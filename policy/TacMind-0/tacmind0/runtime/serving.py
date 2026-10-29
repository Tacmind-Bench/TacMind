"""BaseInferenceConfig: shared TacMind-0 runtime behavior."""

import base64
import binascii
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from io import BytesIO

import numpy as np
import torch
from flask import jsonify, request
from loguru import logger
from PIL import Image
from transformers import AutoProcessor
from werkzeug.exceptions import BadRequest

import tacmind0.data.normalize as normalize
from tacmind0.constants.precision import (
    FP32_MIXED_PRECISION_POLICY,
)
from tacmind0.constants.robot import ROBOT_STATE_DESCS, RobotType
from tacmind0.data.augmentations import (
    NoAugmentationPipeline,
)
from tacmind0.data.transforms import (
    ActionAbsolute,
    ActionRelative,
    ChatTokenization,
    Denormalize,
    Normalize,
    Pipeline,
    PixelTransform,
    ToDevice,
)
from tacmind0.model.modeling import (
    TacMind0ForConditionalGeneration,
)
from tacmind0.model.operations import (
    prepare_action_chunk,
    tacmind0_autocast,
)


@dataclass
class BaseInferenceConfig:
    port: int = 7891
    diffusion_steps: int = 10
    output_action_dim: int = 14
    image_prompts: list[str] = field(
        default_factory=lambda: ["Head", "Left wrist", "Right wrist"]
    )

    @staticmethod
    def _norm_vector_dim(stats: normalize.NormStats, field_name: str) -> int:
        for key in ("q01", "q99", "mean", "std", "min", "max"):
            values = getattr(stats, key)
            if values is not None:
                return int(np.asarray(values).size)
        raise ValueError(f"Cannot infer normalization dimension for {field_name!r}.")

    def _request_default(self, name: str, fallback=None):
        value = getattr(self, name, fallback)
        return fallback if value is None else value

    def _initialize(
        self,
        model: TacMind0ForConditionalGeneration,
        model_name_or_path: str | None,
        norm_stats_path: str,
        n_bins: int,
        model_max_length: int,
        use_absolute_action: bool,
        add_state: bool = True,
    ) -> None:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cpu")

        self.use_absolute_action = use_absolute_action
        self.last_model_latency_sec = None
        self.default_robot_type = self._request_default("default_robot_type")
        self.default_state_desc = self._request_default("default_state_desc")
        self.default_speed = str(self._request_default("default_speed", "0.5"))
        self.default_control_mode = self._request_default("default_control_mode")
        model.eval()
        self.model = model
        model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            config=model.config,
            trust_remote_code=True,
        )
        logger.info("Model loaded successfully")

        self.norm_stats_file = normalize.load_norm_stats_file(norm_stats_path)
        if self.default_robot_type is None and self.norm_stats_file.is_multi_robot:
            self.default_robot_type = self.norm_stats_file.default_robot_type

        input_normalize = Normalize(
            norm_stats_path=norm_stats_path,
            norm_keys=["state"],
            use_quantiles=True,
            norm_stats_file=self.norm_stats_file,
        )
        output_denormalize = Denormalize(
            norm_stats_path=norm_stats_path,
            norm_keys=["action"],
            use_quantiles=True,
            norm_stats_file=self.norm_stats_file,
        )
        self.action_normalize = Normalize(
            norm_stats_path=norm_stats_path,
            norm_keys=["action"],
            use_quantiles=True,
            norm_stats_file=self.norm_stats_file,
        )
        self.action_relative = ActionRelative()
        default_profile = self.norm_stats_file.select(self.default_robot_type)
        self.expected_action_dim = self._norm_vector_dim(
            default_profile["action"],
            "action",
        )
        if self.output_action_dim <= 0:
            raise ValueError(
                f"inference output_action_dim must be positive, got {self.output_action_dim}."
            )
        if self.output_action_dim != self.expected_action_dim:
            raise ValueError(
                "inference output_action_dim must match action normalization "
                f"dimension, got output_action_dim={self.output_action_dim}, "
                f"action_dim={self.expected_action_dim}."
            )
        if self.norm_stats_file.norm_stats_by_robot is not None:
            profile_action_dims = {
                robot_type: self._norm_vector_dim(profile["action"], "action")
                for robot_type, profile in self.norm_stats_file.norm_stats_by_robot.items()
            }
            incompatible_profiles = {
                robot_type: action_dim
                for robot_type, action_dim in profile_action_dims.items()
                if action_dim != self.output_action_dim
            }
            if incompatible_profiles:
                raise ValueError(
                    "All robot profiles served by one inference process must use "
                    f"output_action_dim={self.output_action_dim}; incompatible "
                    f"profiles: {incompatible_profiles}"
                )

        self.input_transform = Pipeline(
            [
                PixelTransform(
                    transform_pipeline=NoAugmentationPipeline(),
                ),
                input_normalize,
                ChatTokenization(
                    processor=self.processor,
                    n_bins=n_bins,
                    max_length=model_max_length,
                    image_prompts=self.image_prompts,
                    add_state=add_state,
                    enable_logging=False,
                ),
                ToDevice(device=self.device),
            ]
        )
        self.output_transform = Pipeline(
            [
                output_denormalize,
                *([ActionAbsolute()] if use_absolute_action else []),
            ]
        )

    def _supported_robot_types(self) -> list[str]:
        supported = [robot_type.value for robot_type in ROBOT_STATE_DESCS]
        default_robot_type = self._request_default("default_robot_type")
        if default_robot_type is not None and default_robot_type not in supported:
            supported.append(default_robot_type)
        return supported

    def _decode_b64_image(
        self, value, field: str, *, require_rgb: bool = False
    ) -> Image.Image:
        if not isinstance(value, str):
            raise ValueError(f"{field} must be a base64 string")
        if value.startswith("data:"):
            if "," not in value:
                raise ValueError(f"{field} data URL is missing base64 payload")
            value = value.split(",", 1)[1]

        try:
            raw = base64.b64decode(value, validate=True)
            image = Image.open(BytesIO(raw))
            if require_rgb and image.mode != "RGB":
                raise ValueError(
                    "Marker image must use RGB mode with three uint8 channels"
                )
            return image.convert("RGB")
        except (binascii.Error, OSError, ValueError) as exc:
            raise ValueError(f"{field} is not a valid base64 image") from exc

    def _parse_state(self, states) -> np.ndarray:
        if states is None:
            raise ValueError("observation.state is required")
        if not isinstance(states, list):
            raise ValueError("observation.state must be a JSON array")

        try:
            state = np.asarray(states, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("observation.state must contain numeric values") from exc
        if state.ndim != 1:
            raise ValueError("observation.state must be a 1D JSON array")
        if state.size == 0:
            raise ValueError("observation.state must not be empty")
        return state

    def _resolve_state_desc(self, robot_type: str | None):
        default_robot_type = self._request_default("default_robot_type")
        default_state_desc = self._request_default("default_state_desc")
        if robot_type is None:
            return default_state_desc

        try:
            robot_type_enum = RobotType(robot_type)
        except ValueError as exc:
            if robot_type == default_robot_type and default_state_desc is not None:
                return default_state_desc
            raise ValueError(
                f"Unsupported robot_type {robot_type!r}. Supported: {self._supported_robot_types()}"
            ) from exc

        if robot_type_enum in ROBOT_STATE_DESCS:
            return ROBOT_STATE_DESCS[robot_type_enum]
        if robot_type == default_robot_type and default_state_desc is not None:
            return default_state_desc
        raise ValueError(
            f"Unsupported robot_type {robot_type!r}. Supported: {self._supported_robot_types()}"
        )

    def _prepare_model_input(
        self,
        *,
        text: str,
        images: list[Image.Image],
        states,
        robot_type: str | None,
        speed: str | None = None,
        control_mode: str | None = None,
    ) -> dict:
        if len(images) != len(self.image_prompts):
            raise ValueError(
                f"Expected {len(self.image_prompts)} images for "
                f"image_prompts={self.image_prompts!r}, got {len(images)}."
            )

        robot_type = (
            self._request_default("default_robot_type")
            if robot_type is None
            else str(robot_type)
        )
        speed = str(
            self._request_default("default_speed", "0.5") if speed is None else speed
        )
        default_control_mode = self._request_default("default_control_mode")
        if control_mode is None:
            control_mode = default_control_mode
        else:
            control_mode = str(control_mode)
        state_desc = self._resolve_state_desc(robot_type)
        if self.use_absolute_action and state_desc is None:
            raise ValueError(
                "observation.robot_type is required unless the dataset configuration "
                "provides state_desc for absolute-action reconstruction."
            )

        state = self._parse_state(states)

        return {
            "images": images,
            "history_images": [],
            "prompt": "" if text is None else str(text),
            "state": state,
            "meta_data": {
                "robot_type": robot_type,
                "speed": speed,
                "control_mode": control_mode,
                "state_desc": state_desc,
            },
        }

    def _predict(self, data: dict) -> np.ndarray:
        self.last_model_latency_sec = None
        state = data["state"]
        model_input = self.input_transform(data)
        inference_seed = os.environ.get("TACMIND0_INFERENCE_SEED")
        if inference_seed is not None:
            seed = int(inference_seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        tacmind0_model = self.model
        action_dim = tacmind0_model.model.config.action_dim
        action_mask = torch.zeros(
            1,
            1,
            action_dim,
            device=self.device,
            dtype=tacmind0_model.model.action_in_proj.weight.dtype,
        )
        action_mask[..., : self.output_action_dim] = 1.0
        inference_kwargs = {
            "input_ids": model_input["input_ids"],
            "attention_mask": model_input["attention_mask"],
            "pixel_values": model_input["pixel_values"],
            "token_type_ids": model_input["token_type_ids"],
            "diffusion_steps": self.diffusion_steps,
            "action_mask": action_mask,
            "history_pixel_values": model_input.get("history_pixel_values"),
            "history_mask": model_input.get("history_mask"),
            **(
                {"tactile_pixel_values": model_input["tactile_pixel_values"]}
                if "tactile_pixel_values" in model_input
                else {}
            ),
        }
        reference_action = data.get("reference_action")
        if reference_action is not None:
            inference_kwargs["reference_action"] = self._prepare_frs_reference_action(
                reference_action,
                reference_action_space=data.get("reference_action_space", "physical"),
                data=data,
                robot_state=state,
                tacmind0_model=tacmind0_model,
                action_mask=action_mask,
            )
        model_t0 = time.perf_counter()
        precision_context = (
            tacmind0_autocast(self.device)
            if tacmind0_model.precision_policy == FP32_MIXED_PRECISION_POLICY
            else nullcontext()
        )
        with precision_context:
            if data.get("sampling_mode") == "tac_frs":
                runtime = getattr(self, "tac_frs_runtime", None)
                if runtime is None:
                    raise ValueError("Checkpoint does not contain Tac-FRS weights")
                with torch.no_grad():
                    actions, _ = runtime.generate(
                        tacmind0_model,
                        inference_kwargs,
                        data["tac_frs_latent"],
                        torch.as_tensor(state, device=self.device).unsqueeze(0),
                    )
            else:
                actions = self.model.inference_action(**inference_kwargs)
        if int(actions.shape[-1]) < self.output_action_dim:
            raise RuntimeError(
                "Model action dimension is smaller than configured output_action_dim: "
                f"model returned {int(actions.shape[-1])}, "
                f"configured {self.output_action_dim}."
            )

        raw_actions = (
            actions.to(torch.float32)
            .detach()
            .cpu()
            .numpy()[0, :, : self.output_action_dim]
        )
        # The CPU copy waits for these actions; avoid stalling unrelated CUDA
        # streams (for example, a simulator sharing the same device).
        self.last_model_latency_sec = time.perf_counter() - model_t0

        outputs = {
            "action": raw_actions,
            "state": state,
            "meta_data": data["meta_data"],
        }
        outputs = self.output_transform(outputs)

        return outputs["action"]

    def _prepare_frs_reference_action(
        self,
        reference_action,
        *,
        reference_action_space: str = "physical",
        data: dict,
        robot_state,
        tacmind0_model,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        chunk_size = int(tacmind0_model.model.config.chunk_size)
        action_dim = int(tacmind0_model.model.config.action_dim)
        dtype = tacmind0_model.model.action_in_proj.weight.dtype
        if reference_action_space not in ("physical", "policy_normalized"):
            raise ValueError(
                "reference_action_space must be physical or policy_normalized"
            )
        if torch.is_tensor(reference_action):
            reference_action = reference_action.detach().float().cpu().numpy()
        arr = np.asarray(reference_action, dtype=np.float32)
        if not np.isfinite(arr).all():
            raise ValueError("reference_action must contain only finite values")
        if reference_action_space == "policy_normalized":
            return prepare_action_chunk(
                torch.as_tensor(arr),
                batch_size=1,
                chunk_size=chunk_size,
                action_dim=action_dim,
                dtype=dtype,
                device=self.device,
                action_mask=action_mask,
            )

        if arr.ndim == 1:
            arr = np.repeat(arr[None, :], chunk_size, axis=0)
        elif arr.ndim != 2:
            raise ValueError(
                "reference_action must have shape [action_dim] or "
                f"[chunk_size, action_dim], got {arr.shape}"
            )
        if arr.shape[-1] != self.output_action_dim:
            raise ValueError(
                "reference_action last dim must match output_action_dim "
                f"{self.output_action_dim}, got {arr.shape[-1]}"
            )
        if arr.shape[0] != chunk_size:
            raise ValueError(
                "reference_action time length must match chunk_size "
                f"{chunk_size}, got {arr.shape[0]}"
            )

        sample = {
            "action": arr,
            "state": np.asarray(robot_state, dtype=np.float32),
            "meta_data": data["meta_data"],
        }
        if self.use_absolute_action:
            sample = self.action_relative(sample)
        sample = self.action_normalize(sample)
        return prepare_action_chunk(
            torch.as_tensor(sample["action"]),
            batch_size=1,
            chunk_size=chunk_size,
            action_dim=action_dim,
            dtype=dtype,
            device=self.device,
            action_mask=action_mask,
        )

    def _apply_v1_sampling(self, sampling):
        if sampling is None:
            return None
        if not isinstance(sampling, dict):
            raise ValueError("sampling must be a JSON object")

        space = sampling.get("reference_action_space", "physical")
        if space not in ("physical", "policy_normalized"):
            raise ValueError(
                "reference_action_space must be physical or policy_normalized"
            )
        if (
            "reference_action_space" in sampling
            and sampling.get("reference_action") is None
        ):
            raise ValueError("reference_action_space requires reference_action")

        num_steps = sampling.get("num_steps")
        if num_steps is not None and int(num_steps) != self.diffusion_steps:
            raise ValueError(
                f"Inference uses fixed num_steps={self.diffusion_steps}, got {num_steps}."
            )

        seed = sampling.get("seed")
        if seed is not None:
            seed = int(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        return sampling.get("reference_action")

    def _parse_image_slot(self, slot) -> int:
        try:
            slot = int(slot)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"image slot must be a 1-based numeric string, got {slot!r}"
            ) from exc

        if slot < 1:
            raise ValueError(f"image slot must be 1-based and positive, got {slot}")
        return slot

    def _prepare_input(self, body: dict) -> dict:
        observation = body.get("observation")
        if not isinstance(observation, dict):
            raise ValueError("observation must be a JSON object")

        images_raw = observation.get("images")
        if not isinstance(images_raw, dict):
            raise ValueError("observation.images must be a JSON object")

        images = []
        sorted_images = sorted(
            images_raw.items(), key=lambda item: self._parse_image_slot(item[0])
        )
        for expected_slot, (slot, image_b64) in enumerate(sorted_images, start=1):
            parsed_slot = self._parse_image_slot(slot)
            if parsed_slot != expected_slot:
                raise ValueError(
                    "observation.images slots must be contiguous 1-based "
                    f"strings; expected {expected_slot}, got {slot!r}"
                )
            images.append(self._decode_b64_image(image_b64, f"images.{slot}"))

        history_images = observation.get("history_images")
        if history_images is not None:
            if not isinstance(history_images, list):
                raise ValueError("observation.history_images must be a JSON array")
            if history_images:
                raise ValueError(
                    "TacMind-0 uses two current visual views; history_images must be empty"
                )

        data = self._prepare_model_input(
            text=observation.get("prompt", ""),
            images=images,
            states=observation.get("state"),
            robot_type=observation.get("robot_type"),
            speed=observation.get("speed"),
            control_mode=observation.get("control_mode"),
        )

        tactile_config = getattr(
            self.model.config, "tactile_config", None
        )
        marker = observation.get("tactile_history")
        if tactile_config is not None:
            from tacmind0.tactile.preprocessing import prepare_history

            if not isinstance(marker, dict) or set(marker) != {"left", "right"}:
                raise ValueError(
                    "observation.tactile_history requires left/right arrays"
                )
            for side in ("left", "right"):
                if not isinstance(marker[side], list) or len(marker[side]) != 8:
                    raise ValueError(
                        "Each tactile_history side requires eight oldest-to-current images"
                    )
            frames = [
                [
                    self._decode_b64_image(
                        value, f"tactile_history.{side}", require_rgb=True
                    )
                    for value in marker[side]
                ]
                for side in ("left", "right")
            ]
            data["tactile_pixel_values"] = prepare_history(*frames).unsqueeze(0)
        elif marker is not None:
            raise ValueError("Tactile input supplied to a vision-only checkpoint")
        return data

    def _infer(self):
        try:
            body = request.get_json(force=True)
            if not isinstance(body, dict):
                raise ValueError("request body must be a JSON object")

            reference_action = self._apply_v1_sampling(body.get("sampling"))
            t0 = time.monotonic()
            data = self._prepare_input(body)
            if reference_action is not None:
                data["reference_action"] = reference_action
                data["reference_action_space"] = body["sampling"].get(
                    "reference_action_space", "physical"
                )
            actions = self._predict(data)
            latency_ms = round((time.monotonic() - t0) * 1000, 1)
            logger.info(
                "v1/infer processing time: {:.3f}s, model_latency_ms={}",
                latency_ms / 1000.0,
                None
                if self.last_model_latency_sec is None
                else round(self.last_model_latency_sec * 1000.0, 3),
            )
            return jsonify(
                {
                    "actions": actions.tolist(),
                    "metadata": {
                        "latency_ms": latency_ms,
                        "sampling_mode": "frs"
                        if reference_action is not None
                        else "without_frs",
                    },
                }
            )
        except (BadRequest, TypeError, ValueError) as exc:
            message = exc.description if isinstance(exc, BadRequest) else str(exc)
            logger.warning(f"v1/infer bad request: {message}")
            return jsonify({"error": message}), 400
