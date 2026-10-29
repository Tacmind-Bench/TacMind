"""RoboColiseum inference for a 30-step delta-joint DM05 policy.

The public service accepts and returns RoboColiseum's native 21-D whole-body
layout.  Internally, only the final torso dimension is predicted in the shared
22-D DM05 layout; the other four torso values are copied from the request.
The default and fast inference backends use a 4096-token multimodal prefix
limit.

Launch with script/dm05_launcher.sh --exp
playground/dm05_robocoliseum_manipulation.py --task inference
--model-config.model-name-or-path /path/to/checkpoint.
"""

import json
from dataclasses import dataclass, field

import numpy as np
import torch
import tyro

from opendm.constants.robot import RobotStateDesc
from opendm.data.transforms import ChatTokenization
from opendm.exp.dm05_exp import DM05InferenceConfig as _CoreDM05InferenceConfig
from opendm.exp.dm05_exp import DM05TrainerConfig as _DM05TrainerConfig
from playground.dm05_robocoliseum import DM05Exp as _RoboColiseumDM05Exp
from playground.dm05_robocoliseum import (
    DM05InferenceConfig as _RoboColiseumDM05InferenceConfig,
)
from playground.dm05_robocoliseum import (
    DM05ModelConfig as _RoboColiseumDM05ModelConfig,
)
from playground.dm05_robocoliseum import RoboColiseumTokenization

# Native request/response order:
# [left joints (7), left gripper, right joints (7), right gripper, torso (5)].
# The model predicts the first 16 native dimensions and the final torso dimension.
# These paired arrays map the predicted dimensions between the two layouts.
PREDICTED_NATIVE_INDICES = np.asarray([*range(16), 20], dtype=np.int64)
NATIVE_TO_SHARED = np.asarray(
    [0, 1, 2, 3, 4, 5, 20, 6, 7, 8, 9, 10, 11, 12, 21, 13, 14],
    dtype=np.int64,
)
VALID_ACTION_DIM_MASK = np.zeros(22, dtype=bool)
VALID_ACTION_DIM_MASK[NATIVE_TO_SHARED] = True
VALID_DIM_MASK = VALID_ACTION_DIM_MASK.copy()
VALID_DIM_MASK[[6, 13]] = False

STATE_DESC: list[RobotStateDesc | str] = [
    RobotStateDesc.JOINT if index in NATIVE_TO_SHARED else "pad" for index in range(22)
]
STATE_DESC[6] = STATE_DESC[13] = RobotStateDesc.GRIPPER
STATE_DESC[14] = "torso"

_NATIVE_TORSO_STATE_KEY = "robocoliseum_native_torso_state"


class RoboColiseumManipulationTokenization(ChatTokenization):
    """Use the checkpoint prompt format without gripper-state text tokens."""

    def __call__(self, data):
        meta = data["meta_data"]
        prompt = data["prompt"]
        text = f"Robot: {meta['robot_type']}\nOverall speed: {meta['speed']}\n"
        if prompt:
            text += f"{prompt}\n"
        content = [{"type": "text", "text": text}]
        for label, image in zip(self.image_prompts, data["images"], strict=True):
            label = f"{label} image: "
            if content[-1]["type"] == "text":
                content[-1]["text"] += label
            else:
                content.append({"type": "text", "text": label})
            content.append({"type": "image", "image": image})
        if self.add_state:
            bins = self.action_to_bin_tokens(data["state"][VALID_DIM_MASK], self.n_bins)
            content.append(
                {
                    "type": "text",
                    "text": "States: " + " ".join(map(str, bins)) + "\n",
                }
            )
        messages = [{"role": "user", "content": content}]

        def apply_chat_template():
            return self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )

        inputs = apply_chat_template()
        if inputs["input_ids"].shape[1] > self.max_length and prompt:
            tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            overflow = inputs["input_ids"].shape[1] - self.max_length
            shortened = self.tokenizer.decode(
                tokens[: max(0, len(tokens) - overflow - 16)],
                skip_special_tokens=False,
            ).strip()
            content[0]["text"] = content[0]["text"].replace(prompt, shortened, 1)
            inputs = apply_chat_template()
        return inputs


@dataclass
class DM05ModelConfig(_RoboColiseumDM05ModelConfig):
    chunk_size: int = field(default=30)


@dataclass
class DM05TrainerConfig(_DM05TrainerConfig):
    model_max_length: int = field(default=4096)


@dataclass
class DM05InferenceConfig(_RoboColiseumDM05InferenceConfig):
    clip_to_bounds: bool = field(default=False)
    # Preserve the checkpoint's FP32 state normalization round trip before
    # applying delta actions.
    use_transformed_state: bool = field(default=True)
    denorm_state: bool = field(default=True)
    fast_prefix_len: int = field(default=4096)
    prefix_seq_len_buckets: list[int] | None = field(
        default_factory=lambda: [896, 1024, 2048, 4096]
    )

    def _request_default_overrides(self) -> dict:
        return {"default_robot_type": "Agibot G2", "default_state_desc": STATE_DESC}

    def _initialize(self, *args, **kwargs) -> None:
        super()._initialize(*args, **kwargs)
        for index, transform in enumerate(self.input_transform.transforms):
            if isinstance(transform, RoboColiseumTokenization):
                self.input_transform.transforms[index] = (
                    RoboColiseumManipulationTokenization(
                        processor=transform.processor,
                        n_bins=transform.n_bins,
                        max_length=transform.max_length,
                        image_prompts=transform.image_prompts,
                        add_state=transform.add_state,
                    )
                )

    def _prepare_model_input(self, **kwargs) -> dict:
        robot_type = str(kwargs.get("robot_type") or "agibot_g2").strip().lower()
        if robot_type not in {"agibot_g2", "agibot g2", "g2", "robocoliseum"}:
            raise ValueError(f"Unsupported robot_type={robot_type!r}")

        states = kwargs["states"]
        if isinstance(states, str):
            states = json.loads(states)
        native_state = np.asarray(states, dtype=np.float32)
        if native_state.shape == (1, 21):
            native_state = native_state[0]
        if native_state.shape != (21,) or not np.isfinite(native_state).all():
            raise ValueError("Expected finite states[21]")

        shared_state = np.zeros(22, dtype=np.float32)
        shared_state[NATIVE_TO_SHARED] = native_state[PREDICTED_NATIVE_INDICES]
        shared_state[[6, 13]] *= -1
        kwargs.update(states=shared_state.tolist(), robot_type="Agibot G2")
        data = _CoreDM05InferenceConfig._prepare_model_input(self, **kwargs)

        prompt = data["prompt"].strip()
        if prompt and prompt[-1] not in ".!?。！？…":
            prompt += "."
        if prompt and not prompt.lower().startswith(("task:", "subtask:")):
            prompt = f"Task: {prompt}"
        data["prompt"] = prompt

        meta = data["meta_data"]
        meta["valid_dim_mask"] = VALID_DIM_MASK.copy()
        meta["valid_action_dim_mask"] = VALID_ACTION_DIM_MASK.copy()
        meta[_NATIVE_TORSO_STATE_KEY] = native_state[16:21].copy()
        return data

    @torch.inference_mode()
    def _predict(self, data: dict) -> np.ndarray:
        shared_actions = _CoreDM05InferenceConfig._predict(self, data)
        native_torso_state = data["meta_data"].get(_NATIVE_TORSO_STATE_KEY)
        if native_torso_state is None:
            raise ValueError("Input 5-D RoboColiseum torso state is required")
        native_torso_state = np.asarray(native_torso_state, dtype=shared_actions.dtype)
        if native_torso_state.shape != (5,):
            raise ValueError(
                "Input RoboColiseum torso state must have shape (5,), "
                f"got {native_torso_state.shape}"
            )

        native_actions = np.empty(
            shared_actions.shape[:-1] + (21,), dtype=shared_actions.dtype
        )
        native_actions[..., PREDICTED_NATIVE_INDICES] = shared_actions[
            ..., NATIVE_TO_SHARED
        ]
        native_actions[..., [7, 15]] *= -1

        torso_prefix = native_torso_state[:4]
        while torso_prefix.ndim < native_actions.ndim:
            torso_prefix = np.expand_dims(torso_prefix, axis=-2)
        native_actions[..., 16:20] = np.broadcast_to(
            torso_prefix, native_actions.shape[:-1] + (4,)
        )
        return native_actions


@dataclass
class DM05Exp(_RoboColiseumDM05Exp):
    model_config: DM05ModelConfig = field(default_factory=DM05ModelConfig)
    trainer_config: DM05TrainerConfig = field(default_factory=DM05TrainerConfig)
    inference_config: DM05InferenceConfig = field(default_factory=DM05InferenceConfig)


if __name__ == "__main__":
    tyro.cli(DM05Exp).inference()
