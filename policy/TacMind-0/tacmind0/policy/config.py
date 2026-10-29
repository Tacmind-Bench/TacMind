"""TacMind-0 model, data, and trainer configuration."""

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from torch import nn

from tacmind0.constants.robot import ActionMode, RobotStateDesc, RobotType
from tacmind0.data.augmentations import (
    NoAugmentationPipeline,
    TrainingTransformPipeline,
)
from tacmind0.data.collator import TrainingCollator
from tacmind0.data.dataset import TacMind0Dataset
from tacmind0.data.transforms import (
    BuildAction,
    ChatTokenization,
    LoadImages,
    Normalize,
    PadAction,
    Pipeline,
    PixelTransform,
)
from tacmind0.model.modeling import TacMind0Config
from tacmind0.runtime.data import compute_norm_stats
from tacmind0.runtime.model import BaseModelConfig
from tacmind0.runtime.training import BaseTrainerConfig
from tacmind0.tactile.encoder import FrozenTactileEncoder
from tacmind0.tactile.preprocessing import LoadTactileHistory

from .checkpoints import strict_policy_load
from .contracts import CONTRACT, validate_contract, validate_visual_augmentation
from .paths import ARTIFACTS_ROOT, PROJECT_ROOT


@dataclass
class TacMind0ModelConfig(BaseModelConfig):
    model_name_or_path: str = str(ARTIFACTS_ROOT / "base_policy")
    chunk_size: int = 32
    tactile_weights: str = str(ARTIFACTS_ROOT / "taclewm" / "encoder.pt")
    backbone_config: str = str(ARTIFACTS_ROOT / "taclewm" / "backbone_config.json")
    replace_tactile_encoder: bool = False
    vision_attn_implementation: str = "sdpa"
    liger_kernel: bool = False

    def _load_base_checkpoint_model(self):
        saved = TacMind0Config.from_pretrained(
            self.model_name_or_path, local_files_only=True
        )
        if saved.tactile_config is not None:
            validate_contract(saved.tactile_config)
            self.chunk_size = saved.chunk_size
        attach_encoder = saved.tactile_config is None or self.replace_tactile_encoder
        model = strict_policy_load(
            self.model_name_or_path,
            dtype=self._torch_dtype(),
            overrides={"chunk_size": self.chunk_size},
        )
        if attach_encoder:
            backbone = json.loads(Path(self.backbone_config).read_text())
            encoder = FrozenTactileEncoder(backbone)
            report = encoder.load_world_model(self.tactile_weights)
            # Keep a trained policy's action/observation contract and projection
            # weights when explicitly installing a newly adapted encoder.
            new_policy = model.config.tactile_config is None
            contract = CONTRACT if new_policy else model.config.tactile_config
            model.config.tactile_config = {**deepcopy(contract), "backbone": backbone}
            model.model.config.tactile_config = model.config.tactile_config
            model.tactile_encoder = encoder.to(dtype=self._torch_dtype())
            if new_policy:
                model.tactile_projector = nn.Linear(
                    384, model.config.vlm_config.text_config.hidden_size
                ).to(dtype=self._torch_dtype())
            model.tactile_load_report = report
        return model

    def build_model(self):
        if (Path(self.model_name_or_path) / "adapter_config.json").is_file():
            raise ValueError("TacMind-0 requires full-policy training, not LoRA")
        return super().build_model()


@dataclass
class TacMind0DataConfig:
    marker_cache_size: int = 256
    jsonl_dir: str = str(PROJECT_ROOT / "data" / "episodes")
    image_dir: str = str(PROJECT_ROOT / "data")
    norm_file: str = str(ARTIFACTS_ROOT / "base_policy" / "norm_stats.json")
    n_bins: int = 256
    compute_norm_stats_max_batches: int | None = None
    action_mode: ActionMode = ActionMode.RELATIVE
    add_state: bool = True
    state_desc: list[str] = field(default_factory=lambda: list(CONTRACT["state_desc"]))
    image_prompts: list[str] = field(
        default_factory=lambda: list(CONTRACT["image_prompts"])
    )
    agentview_augmentation: Literal["none", "light"] = "none"
    wrist_augmentation: Literal["none", "light"] = "none"
    augmentation_probability: float = 0.5

    def visual_augmentation_contract(self) -> dict:
        value = {
            "agentview": self.agentview_augmentation,
            "wrist": self.wrist_augmentation,
            "probability": self.augmentation_probability,
        }
        validate_visual_augmentation(value)
        return value

    def dataset_metadata(self):
        return {
            "robot_type": RobotType.FRANKA,
            "state_desc": [RobotStateDesc(value) for value in self.state_desc],
        }

    def norm_stats_path(self, action_horizon):
        return Path(self.norm_file)

    def _action_transform(self, action_horizon):
        return BuildAction(action_horizon=action_horizon, action_mode=self.action_mode)

    def build_norm_stats_dataset(self, action_horizon):
        return TacMind0Dataset(
            self.jsonl_dir,
            Pipeline([self._action_transform(action_horizon)]),
            dataset_meta=self.dataset_metadata(),
        )

    def build_dataset(self, processor, action_horizon, tokenizer_max_length=1024):
        augmentation = self.visual_augmentation_contract()
        transforms = Pipeline(
            [
                LoadTactileHistory(self.image_dir, cache_size=self.marker_cache_size),
                self._action_transform(action_horizon),
                LoadImages(["images_1", "images_2"], self.image_dir),
                PixelTransform(
                    image_pipelines=[
                        (
                            TrainingTransformPipeline(p=augmentation["probability"])
                            if mode == "light"
                            else NoAugmentationPipeline()
                        )
                        for mode in (
                            augmentation["agentview"],
                            augmentation["wrist"],
                        )
                    ]
                ),
                Normalize(
                    str(self.norm_stats_path(action_horizon)),
                    ["state", "action"],
                    use_quantiles=True,
                ),
                ChatTokenization(
                    processor,
                    n_bins=self.n_bins,
                    max_length=tokenizer_max_length,
                    image_prompts=list(self.image_prompts),
                    add_state=self.add_state,
                ),
                PadAction(32),
            ]
        )
        return (
            TacMind0Dataset(
                self.jsonl_dir, transforms, dataset_meta=self.dataset_metadata()
            ),
            TrainingCollator(processor.tokenizer.pad_token_id, tokenizer_max_length),
        )

    def compute_norm_stats(self, action_horizon, batch_size=128, num_workers=32):
        compute_norm_stats(
            self.build_norm_stats_dataset(action_horizon),
            self.norm_stats_path(action_horizon),
            batch_size=batch_size,
            num_workers=num_workers,
            max_batches=self.compute_norm_stats_max_batches,
        )


@dataclass
class TacMind0TrainerConfig(BaseTrainerConfig):
    output_dir: str = str(PROJECT_ROOT / "user_checkpoints" / "tacmind0")
    fsdp1: bool = False
    per_device_train_batch_size: int = 1
