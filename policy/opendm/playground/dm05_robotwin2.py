"""RoboTwin2 DM05 SFT with native 14D normalization and 32D model actions."""

import copy
import os
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

import numpy as np
from torch.utils.data import ConcatDataset

from opendm.constants.robot import ROBOT_STATE_DESCS, ActionMode, RobotType
from opendm.data.augmentations import NoAugmentationPipeline
from opendm.data.collator import TrainingCollator
from opendm.data.dataset import JsonlDataset
from opendm.data.transforms import (
    BuildAction,
    ChatTokenization,
    LoadImages,
    Normalize,
    PadAction,
    Pipeline,
    PixelTransform,
)
from opendm.dataset.register import CONVERSATION_DATA
from opendm.dataset.robotwin2 import (
    ROBOTWIN2_COMPONENT_SETTINGS,
    ROBOTWIN2_COMPONENT_TASKS,
)
from opendm.exp.dm05_exp import DM05DataConfig as _DM05DataConfig
from opendm.exp.dm05_exp import DM05Exp as _DM05Exp
from opendm.exp.dm05_exp import DM05InferenceConfig as _DM05InferenceConfig
from opendm.exp.dm05_exp import DM05OptimizerConfig as _DM05OptimizerConfig
from opendm.exp.dm05_exp import DM05TrainerConfig as _DM05TrainerConfig

_ROBOTWIN2_CLEAN_WEIGHT = 0.2
_ROBOTWIN2_CRITICAL_LOW_TASK_WEIGHT = 1.8
_ROBOTWIN2_WATCH_LOW_TASK_WEIGHT = 1.25
_ROBOTWIN2_CRITICAL_LOW_SUCCESS_TASKS = {
    "hanging_mug",
    "move_stapler_pad",
    "turn_switch",
    "place_dual_shoes",
    "scan_object",
    "place_mouse_pad",
    "beat_block_hammer",
    "stack_bowls_three",
}
_ROBOTWIN2_WATCH_LOW_SUCCESS_TASKS = {
    "blocks_ranking_size",
    "place_can_basket",
    "stamp_seal",
    "put_object_cabinet",
    "place_bread_skillet",
    "put_bottles_dustbin",
    "place_object_basket",
    "pick_diverse_bottles",
}


def _robotwin2_component_dataset_name(task: str, setting: str) -> str:
    return f"robotwin2_component_{task}_{setting}"


def _robotwin2_legacy_dataset_name(task: str, setting: str) -> str:
    return f"robotwin2_example_{task}_{setting}"


def _robotwin2_dataset_weight(task: str, setting: str) -> float:
    weight = _ROBOTWIN2_CLEAN_WEIGHT if setting == "clean" else 1.0
    if task in _ROBOTWIN2_CRITICAL_LOW_SUCCESS_TASKS:
        weight *= _ROBOTWIN2_CRITICAL_LOW_TASK_WEIGHT
    elif task in _ROBOTWIN2_WATCH_LOW_SUCCESS_TASKS:
        weight *= _ROBOTWIN2_WATCH_LOW_TASK_WEIGHT
    return weight


_COMPONENT_DATASET_NAMES = [
    _robotwin2_component_dataset_name(task, setting)
    for task in ROBOTWIN2_COMPONENT_TASKS
    for setting in ROBOTWIN2_COMPONENT_SETTINGS
]
_COMPONENT_DATASET_WEIGHTS = [
    _robotwin2_dataset_weight(task, setting)
    for task in ROBOTWIN2_COMPONENT_TASKS
    for setting in ROBOTWIN2_COMPONENT_SETTINGS
]
_COMPONENT_DATASET_SUBDIRS = {
    _robotwin2_component_dataset_name(task, setting): f"{task}/{setting}"
    for task in ROBOTWIN2_COMPONENT_TASKS
    for setting in ROBOTWIN2_COMPONENT_SETTINGS
}
_COMPONENT_DATASET_SUBDIRS.update(
    {
        _robotwin2_legacy_dataset_name(task, setting): f"{task}/{setting}"
        for task in ROBOTWIN2_COMPONENT_TASKS
        for setting in ROBOTWIN2_COMPONENT_SETTINGS
    }
)


class PrepareRobotwin2Actions:
    """Validate native 14D state/actions and convert them to float32."""

    native_dim = 14

    def __call__(self, data, **kwargs):
        del kwargs
        for key in ("state", "action"):
            value = np.asarray(data[key], dtype=np.float32)
            if value.ndim == 0 or value.shape[-1] != self.native_dim:
                raise ValueError(
                    f"RoboTwin2 {key} must have 14 dimensions, got {value.shape}"
                )
            data[key] = value
        return data


def _refine_prompt(text) -> str:
    refined = "" if text is None else str(text).strip()
    if refined and refined[-1] not in ".!?。！？…":
        refined += "."
    return refined


class SampleRobotwin2TargetPrompt:
    """Match the target recipe's prompt sampling and RNG consumption."""

    @staticmethod
    def _candidates(value) -> list:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    def __call__(self, data, **kwargs):
        del kwargs
        original = data.get("prompt")
        subtask = data.get("subtask")
        # SamplePrompt consumes this draw even when subtask is unavailable.
        add_original = random.random() < 0.0 or subtask is None
        prompt = ""
        if add_original:
            candidates = [original, *self._candidates(data.get("rewrite_prompt"))]
            candidates = [item for item in candidates if item is not None]
            prompt += f"Task: {_refine_prompt(random.choice(candidates))}\n"
        if subtask is not None:
            candidates = [subtask, *self._candidates(data.get("subtask_rewrites"))]
            candidates = [item for item in candidates if item is not None]
            prompt += f"Subtask: {random.choice(candidates)}\n"
        data["prompt"] = _refine_prompt(prompt or original)
        return data


@dataclass
class DM05DataConfig(_DM05DataConfig):
    dataset_name: str = field(default="+".join(_COMPONENT_DATASET_NAMES))
    dataset_weights: list[float] | None = field(
        default_factory=lambda: list(_COMPONENT_DATASET_WEIGHTS)
    )
    num_train_samples: int | None = field(default=None)
    action_mode: ActionMode = field(default=ActionMode.ABSOLUTE)
    norm_stats_root: str = field(default="./norm_stats/robotwin2_absolute_action/14d")

    def _dataset_info_for(self, name: str) -> dict:
        if name not in CONVERSATION_DATA:
            raise KeyError(f"Unknown dataset: {name}")
        info = {
            key: value.value if isinstance(value, Enum) else copy.deepcopy(value)
            for key, value in CONVERSATION_DATA[name].items()
        }
        if self.jsonl_dir is not None:
            component_subdir = _COMPONENT_DATASET_SUBDIRS.get(name)
            info["jsonl_dir"] = (
                f"{self.jsonl_dir.rstrip('/')}/{component_subdir}/"
                if component_subdir is not None
                else self.jsonl_dir
            )
        if self.image_dir is not None:
            info["image_dir"] = self.image_dir
        return info

    def _dataset_info(self) -> dict:
        # All RoboTwin2 components share the robot metadata used by inference.
        return self._dataset_info_for(self.dataset_name.split("+")[0])

    def _action_transform(self, action_horizon: int) -> BuildAction:
        return BuildAction(
            action_horizon=action_horizon,
            action_mode=ActionMode.ABSOLUTE,
            action_source="state",
        )

    def build_norm_stats_dataset(self, action_horizon: int) -> ConcatDataset:
        datasets = []
        for name in self.dataset_name.split("+"):
            info = self._dataset_info_for(name)
            datasets.append(
                JsonlDataset(
                    jsonl_dir=info["jsonl_dir"],
                    transforms=Pipeline(
                        [
                            self._action_transform(action_horizon),
                            PrepareRobotwin2Actions(),
                        ]
                    ),
                    dataset_name=name,
                    dataset_meta=self._dataset_meta(info),
                )
            )
        return ConcatDataset(datasets)

    def norm_stats_path(self, action_horizon: int) -> Path:
        del action_horizon
        return Path(self.norm_stats_root) / "norm_stats.json"

    def _build_component_dataset(
        self,
        name: str,
        processor,
        action_horizon: int,
        tokenizer_max_length: int,
    ) -> JsonlDataset:
        dataset_info = self._dataset_info_for(name)
        pipeline = Pipeline(
            [
                self._action_transform(action_horizon),
                PrepareRobotwin2Actions(),
                Normalize(
                    norm_stats_path=str(self.norm_stats_path(action_horizon)),
                    norm_keys=["state", "action"],
                    use_quantiles=True,
                    clip_to_bounds=False,
                ),
                LoadImages(
                    image_keys=dataset_info["image_keys"],
                    image_dir=dataset_info["image_dir"],
                ),
                PixelTransform(transform_pipeline=NoAugmentationPipeline()),
                SampleRobotwin2TargetPrompt(),
                ChatTokenization(
                    processor=processor,
                    n_bins=self.n_bins,
                    max_length=tokenizer_max_length,
                    image_prompts=dataset_info["image_prompts"],
                    add_state=self.add_state,
                    prompt_is_formatted=True,
                ),
                PadAction(32),
            ]
        )
        dataset_meta = {
            key: value
            for key, value in dataset_info.items()
            if key not in {"jsonl_dir", "image_dir", "image_keys", "image_prompts"}
        }
        return JsonlDataset(
            jsonl_dir=dataset_info["jsonl_dir"],
            transforms=pipeline,
            dataset_name=name,
            dataset_meta=dataset_meta,
            drop_last_frame=False,
            tail_resample_count=2,
            tail_global_resample_count=3,
        )

    def build_dataset(
        self,
        processor,
        action_horizon: int,
        tokenizer_max_length: int = 1152,
    ) -> tuple:
        names = self.dataset_name.split("+")
        if self.dataset_weights is None or len(self.dataset_weights) != len(names):
            raise ValueError("dataset_weights must match component datasets")
        datasets = [
            self._build_component_dataset(
                name, processor, action_horizon, tokenizer_max_length
            )
            for name in names
        ]
        collator = TrainingCollator(
            pad_token_id=processor.tokenizer.pad_token_id,
            max_length=tokenizer_max_length,
        )
        return ConcatDataset(datasets), collator


@dataclass
class DM05OptimizerConfig(_DM05OptimizerConfig):
    base_lr: float = field(default=4e-5)
    optim: Literal["adamw", "muon_adamw"] = field(default="muon_adamw")


@dataclass
class DM05TrainerConfig(_DM05TrainerConfig):
    output_dir: str = field(
        default=f"user_checkpoints/{os.path.basename(__file__)[:-3]}"
    )
    per_device_train_batch_size: int = field(default=32)
    save_steps: int = field(default=10000)
    save_total_limit: int = field(default=100)
    model_max_length: int = field(default=1152)
    wandb_project: str | None = field(default="dm05_sft")


@dataclass
class DM05InferenceConfig(_DM05InferenceConfig):
    clip_to_bounds: bool = field(default=False)

    def _request_default_overrides(self) -> dict:
        return {
            "default_robot_type": RobotType.ALOHA_ROBOTWIN2.value,
            "default_state_desc": list(ROBOT_STATE_DESCS[RobotType.ALOHA_ROBOTWIN2]),
        }


@dataclass
class DM05Exp(_DM05Exp):
    task: Literal["train", "inference", "compute_norm_stats"] = field(default="train")
    optimizer_config: DM05OptimizerConfig = field(default_factory=DM05OptimizerConfig)
    trainer_config: DM05TrainerConfig = field(default_factory=DM05TrainerConfig)
    data_config: DM05DataConfig = field(default_factory=DM05DataConfig)
    inference_config: DM05InferenceConfig = field(default_factory=DM05InferenceConfig)

    def _initialize_inference_runtime(self) -> None:
        if self.data_config.action_mode != ActionMode.ABSOLUTE:
            raise ValueError("RoboTwin2 inference requires absolute actions")
        super()._initialize_inference_runtime()


if __name__ == "__main__":
    import tyro

    exp = tyro.cli(DM05Exp)
    if exp.task == "train":
        exp.train()
    elif exp.task == "inference":
        exp.inference()
    elif exp.task == "compute_norm_stats":
        exp.data_config.compute_norm_stats(action_horizon=exp.model_config.chunk_size)
    else:
        raise ValueError(f"Invalid task: {exp.task}")
