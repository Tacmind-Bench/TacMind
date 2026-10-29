"""Training and serving lifecycle; concrete policies supply their data config."""

import os
import pathlib
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from flask import Flask
from loguru import logger
from transformers import AutoProcessor

from tacmind0.constants.robot import ActionMode
from tacmind0.runtime.trainer import PolicyTrainer, safe_save_model_for_hf_trainer

from .model import BaseModelConfig
from .optimization import OptimizerConfig
from .serving import BaseInferenceConfig
from .training import BaseTrainerConfig


@dataclass
class BaseExperiment:
    task: Literal["train", "inference"] = "train"
    model_config: BaseModelConfig = field(default_factory=BaseModelConfig)
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    trainer_config: BaseTrainerConfig = field(default_factory=BaseTrainerConfig)
    inference_config: BaseInferenceConfig = field(default_factory=BaseInferenceConfig)
    logger_level: str = "INFO"

    def __post_init__(self):
        logger.remove()
        logger.add(sys.stdout, level=self.logger_level)

    def _ensure_norm_stats(self) -> None:
        norm_stats_path = self.data_config.norm_stats_path(self.model_config.chunk_size)
        if norm_stats_path.exists():
            logger.info(f"Using norm stats from {norm_stats_path}")
            return

        if self.local_rank == 0:
            self.data_config.compute_norm_stats(
                action_horizon=self.model_config.chunk_size
            )
        else:
            while not norm_stats_path.exists():
                time.sleep(5)
        assert norm_stats_path.exists()

    def _initialize_train(self):
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        logger.info(f"Local rank: {self.local_rank}")
        if self.local_rank != 0:
            logger.remove()
            logger.add(lambda msg: None)

        self._ensure_norm_stats()

        model = self.model_config.build_model()
        self.model = model
        self.optimizer_config.prepare_model(model)
        model.config.use_cache = False
        model.model.vlm.config.use_cache = False

        processor = AutoProcessor.from_pretrained(
            self.model_config.model_name_or_path,
            config=model.config,
            trust_remote_code=True,
        )
        self.processor = processor
        self.tokenizer = processor.tokenizer

        train_dataset, data_collator = self.data_config.build_dataset(
            processor=processor,
            action_horizon=self.model_config.chunk_size,
            tokenizer_max_length=self.trainer_config.model_max_length,
        )

        trainer_kwargs = {
            "model": self.model,
            "processing_class": self.processor,
            "exp_config": self,
            "train_dataset": train_dataset,
            "data_collator": data_collator,
        }
        trainer = PolicyTrainer(**trainer_kwargs)
        self.trainer = trainer

    def train(self):
        self._initialize_train()

        if list(pathlib.Path(self.trainer_config.output_dir).glob("checkpoint-*")):
            self.trainer.train(resume_from_checkpoint=True)
        else:
            self.trainer.train()

        self.trainer.save_state()
        self.model.config.use_cache = True
        self.model.model.vlm.config.use_cache = True
        safe_save_model_for_hf_trainer(
            trainer=self.trainer,
            output_dir=self.trainer_config.output_dir,
        )
        logger.info(
            f"Training completed and model saved to {self.trainer_config.output_dir}"
        )

    def _initialize_inference_runtime(self) -> None:
        """Build the model and transforms shared by inference entry points."""
        logger.info(f"Loading model from {self.model_config.model_name_or_path}")
        model = self.model_config.build_model()
        ckpt_norm_stats_path = (
            pathlib.Path(self.model_config.model_name_or_path) / "norm_stats.json"
        )
        norm_stats_path = (
            ckpt_norm_stats_path
            if ckpt_norm_stats_path.exists()
            else self.data_config.norm_stats_path(self.model_config.chunk_size)
        )
        if ckpt_norm_stats_path.exists():
            logger.info(f"Using normalization stats from checkpoint: {norm_stats_path}")
        else:
            logger.warning(
                "Normalization stats not found in checkpoint at "
                f"{ckpt_norm_stats_path}; falling back to {norm_stats_path}"
            )
        metadata = self.data_config.dataset_metadata()
        default_robot_type = metadata.get("robot_type")
        if isinstance(default_robot_type, Enum):
            default_robot_type = default_robot_type.value
        self.inference_config.default_robot_type = default_robot_type
        self.inference_config.default_state_desc = metadata.get("state_desc")
        self.inference_config._initialize(
            model=model,
            model_name_or_path=self.model_config.model_name_or_path,
            norm_stats_path=str(norm_stats_path),
            n_bins=self.data_config.n_bins,
            model_max_length=self.trainer_config.model_max_length,
            use_absolute_action=(self.data_config.action_mode == ActionMode.RELATIVE),
            add_state=self.data_config.add_state,
        )

    def inference(self) -> None:
        self._initialize_inference_runtime()
        app = Flask(__name__)
        app.add_url_rule(
            "/v1/infer",
            "v1_infer",
            self.inference_config._infer,
            methods=["POST"],
        )
        if hasattr(self.inference_config, "_release_tac_frs"):
            app.add_url_rule(
                "/v1/tac_frs/release",
                "tac_frs_release",
                self.inference_config._release_tac_frs,
                methods=["POST"],
            )
        app.run(
            host="0.0.0.0",
            port=self.inference_config.port,
            debug=False,
            threaded=False,
        )
