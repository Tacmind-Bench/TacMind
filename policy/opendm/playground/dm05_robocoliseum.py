"""RoboColiseum G2 inference entrypoint for DM05.

Launch with script/dm05_launcher.sh --exp playground/dm05_robocoliseum.py
--task inference --model-config.model-name-or-path /path/to/checkpoint.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro
from flask import Flask, jsonify

from opendm.constants.robot import RobotStateDesc
from opendm.data.transforms import ChatTokenization
from opendm.exp.dm05_exp import DM05Exp as _DM05Exp
from opendm.exp.dm05_exp import DM05InferenceConfig as _DM05InferenceConfig
from opendm.exp.dm05_exp import DM05ModelConfig as _DM05ModelConfig

# RoboColiseum native layout:
# [left joints (7), left gripper, right joints (7), right gripper].
# The DM05 shared layout stores the seventh joints at indices 20 and 21.
NATIVE_TO_SHARED = np.asarray([0, 1, 2, 3, 4, 5, 20, 6, 7, 8, 9, 10, 11, 12, 21, 13])
VALID_DIM_MASK = np.zeros(22, dtype=bool)
VALID_DIM_MASK[NATIVE_TO_SHARED] = True
STATE_DESC = [RobotStateDesc.JOINT] * 22
STATE_DESC[6] = STATE_DESC[13] = RobotStateDesc.GRIPPER


class RoboColiseumTokenization(ChatTokenization):
    """Checkpoint prompt format and state tokens in valid shared-dimension order."""

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
                {"type": "text", "text": "States: " + " ".join(map(str, bins)) + "\n"}
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
class DM05ModelConfig(_DM05ModelConfig):
    chunk_size: int = field(default=50)
    precision_policy: Literal["bf16_mixed", "fp32_mixed"] = field(default="fp32_mixed")
    llm_attn_implementation: Literal["auto", "eager", "sdpa", "flex_attention"] = field(
        default="sdpa"
    )
    vision_attn_implementation: Literal[
        "auto", "eager", "sdpa", "flash_attention_2"
    ] = field(default="sdpa")
    liger_kernel: bool = field(default=False)


@dataclass
class DM05InferenceConfig(_DM05InferenceConfig):
    # The base pipeline normalizes and reconstructs actions in shared space.
    output_action_dim: int = field(default=22)

    def _request_default_overrides(self) -> dict:
        return {"default_robot_type": "Agibot G2", "default_state_desc": STATE_DESC}

    def _initialize(self, *args, **kwargs) -> None:
        if kwargs.get("is_history", False):
            raise ValueError("RoboColiseum G2 does not use history images")
        if self.output_action_dim != 22 or len(self.image_prompts) != 3:
            raise ValueError(
                "RoboColiseum G2 requires 22 shared dimensions and 3 cameras"
            )
        super()._initialize(*args, **kwargs)
        for index, transform in enumerate(self.input_transform.transforms):
            if isinstance(transform, ChatTokenization):
                self.input_transform.transforms[index] = RoboColiseumTokenization(
                    processor=transform.processor,
                    n_bins=transform.n_bins,
                    max_length=transform.max_length,
                    image_prompts=transform.image_prompts,
                    add_state=transform.add_state,
                )

    def _prepare_model_input(self, **kwargs) -> dict:
        robot_type = str(kwargs.get("robot_type") or "agibot_g2").strip().lower()
        if robot_type not in {"agibot_g2", "agibot g2", "g2", "robocoliseum"}:
            raise ValueError(f"Unsupported robot_type={robot_type!r}")
        states = kwargs["states"]
        if isinstance(states, str):
            states = json.loads(states)
        native_state = np.asarray(states, dtype=np.float32)
        if native_state.shape == (1, 16):
            native_state = native_state[0]
        if native_state.shape != (16,) or not np.isfinite(native_state).all():
            raise ValueError("Expected finite states[16]")
        shared_state = np.zeros(22, dtype=np.float32)
        shared_state[NATIVE_TO_SHARED] = native_state
        kwargs.update(states=shared_state.tolist(), robot_type="Agibot G2")
        data = super()._prepare_model_input(**kwargs)
        prompt = data["prompt"].strip()
        if prompt and prompt[-1] not in ".!?。！？…":
            prompt += "."
        if prompt and not prompt.lower().startswith(("task:", "subtask:")):
            prompt = f"Task: {prompt}"
        data["prompt"] = prompt
        data["meta_data"]["valid_action_dim_mask"] = VALID_DIM_MASK.copy()
        return data

    @torch.inference_mode()
    def _predict(self, data: dict) -> np.ndarray:
        return super()._predict(data)[..., NATIVE_TO_SHARED]


@dataclass
class DM05Exp(_DM05Exp):
    task: Literal["inference"] = field(default="inference")
    use_lora: bool | None = field(default=False)
    norm_stats: str | None = field(default=None)
    model_config: DM05ModelConfig = field(default_factory=DM05ModelConfig)
    inference_config: DM05InferenceConfig = field(default_factory=DM05InferenceConfig)

    def _initialize_inference_runtime(self) -> None:
        # Build the core model and transform pipeline directly; this
        # inference-only entrypoint does not require dataset registration.
        checkpoint = self.model_config.model_name_or_path
        norm_stats = self.norm_stats or str(Path(checkpoint) / "norm_stats.json")
        if self.inference_config.backend == "fast":
            self.model_config.llm_attn_implementation = "flex_attention"
        model = self.model_config.build_model(use_lora=self.use_lora)
        self.inference_config._initialize(
            model=model,
            model_name_or_path=checkpoint,
            norm_stats_path=norm_stats,
            n_bins=self.data_config.n_bins,
            model_max_length=self.trainer_config.model_max_length,
            use_absolute_action=True,
            add_state=self.data_config.add_state,
            is_history=self.data_config.is_history,
        )

    def create_app(self) -> Flask:
        app = Flask(__name__)
        config = self.inference_config
        app.add_url_rule(
            "/process_frame", "process_frame", config._infer_legacy, methods=["POST"]
        )
        app.add_url_rule("/v1/infer", "v1_infer", config._infer, methods=["POST"])

        @app.get("/healthz")
        @app.get("/readyz")
        def health():
            return jsonify(
                status="ready", robot_type="agibot_g2", backend=config.backend
            )

        return app

    def inference(self) -> None:
        self._initialize_inference_runtime()
        self.create_app().run(
            host="0.0.0.0", port=self.inference_config.port, debug=False, threaded=False
        )


if __name__ == "__main__":
    tyro.cli(DM05Exp).inference()
