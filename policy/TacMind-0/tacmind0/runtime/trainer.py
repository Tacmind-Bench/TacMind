import os
import shutil
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from transformers import Trainer, TrainingArguments

from tacmind0.constants.precision import (
    FP32_MIXED_PRECISION_POLICY,
    FSDP_MIXED_PRECISION,
    TF32_ENABLED,
)


class PolicyTrainer(Trainer):
    """Trainer extension for TacMind0 model training."""

    _FSDP_WRAP_ATTR = "_tacmind0_fsdp_wrap"

    def __init__(self, *args, **kwargs):
        self.exp_config: Any = kwargs.pop("exp_config")
        training_args = self._link_exp_config()
        super().__init__(*args, args=training_args, **kwargs)
        self.loss_cache = {}

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        target = self.model if model is None else model
        if getattr(target.config, "tactile_config", None) is not None:
            # from_pretrained applies backbone key conversions that Trainer's raw
            # state_dict loader bypasses. The experiment loads the last checkpoint
            # before creating Trainer; optimizer/scheduler/RNG restore separately.
            logger.info("TacMind0 model loaded before Trainer initialization")
            return
        return super()._load_from_checkpoint(resume_from_checkpoint, model=model)

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict=state_dict)
        destination = Path(output_dir or self.args.output_dir)
        norm_path = self.exp_config.data_config.norm_stats_path(
            self.exp_config.model_config.chunk_size
        )
        if Path(norm_path).resolve() != (destination / "norm_stats.json").resolve():
            shutil.copyfile(norm_path, destination / "norm_stats.json")
        if getattr(self.model.config, "tactile_config", None) is not None:
            import json
            from dataclasses import fields, is_dataclass
            from enum import Enum

            def serialize(value):
                """Encode configuration fields; runtime attributes are excluded."""
                if isinstance(value, Enum):
                    return value.value
                if isinstance(value, Path):
                    return str(value)
                if is_dataclass(value):
                    return {
                        field.name: getattr(value, field.name)
                        for field in fields(value)
                    }
                raise TypeError(
                    f"Unsupported configuration value: {type(value).__name__}"
                )

            (destination / "training_config.json").write_text(
                json.dumps(self.exp_config, default=serialize, indent=2)
            )

    def create_accelerator_and_postprocess(self):
        super().create_accelerator_and_postprocess()
        self._configure_fsdp_precision()
        self._configure_fsdp_auto_wrap()

    @classmethod
    def _fsdp_auto_wrap_policy(
        cls,
        module: torch.nn.Module,
        recurse: bool,
        nonwrapped_numel: int,
    ) -> bool:
        if recurse:
            return True
        return bool(getattr(module, cls._FSDP_WRAP_ATTR, False))

    def _configure_fsdp_precision(self) -> None:
        if self.exp_config.model_config.precision_policy != FP32_MIXED_PRECISION_POLICY:
            return
        fsdp_plugin = self.accelerator.state.fsdp_plugin
        if fsdp_plugin is not None:
            fsdp_plugin.set_mixed_precision(
                FSDP_MIXED_PRECISION,
                override=True,
            )
            logger.info(
                "TacMind0 FSDP mixed precision policy: {}",
                FSDP_MIXED_PRECISION,
            )

    def _configure_fsdp_auto_wrap(self) -> None:
        if self.exp_config.model_config.precision_policy != FP32_MIXED_PRECISION_POLICY:
            return
        fsdp_plugin = self.accelerator.state.fsdp_plugin
        if fsdp_plugin is None:
            return

        modules = self.model.fsdp_wrap_modules()
        for module in modules:
            setattr(module, self._FSDP_WRAP_ATTR, True)

        fsdp_plugin.auto_wrap_policy = self._fsdp_auto_wrap_policy
        fsdp_plugin.transformer_cls_names_to_wrap = None
        fsdp_plugin.min_num_params = 0
        logger.info(
            "TacMind0 FSDP auto-wrap policy: action expert and final {} VLM layers",
            len(modules) - 1,
        )

    def create_optimizer(self, model=None) -> torch.optim.Optimizer:
        opt_model: torch.nn.Module = model if model is not None else self.model

        if self.optimizer is None:
            if self.exp_config.optimizer_config.optim == "muon_adamw":
                self.optimizer = self.exp_config.optimizer_config.build_muon_adamw(
                    opt_model
                )
                return self.optimizer
            optimizer_grouped_parameters = (
                self.exp_config.optimizer_config._get_optimizer_grouped_parameters(
                    opt_model
                )
            )

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )
            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )

        return self.optimizer

    def _load_optimizer_and_scheduler(self, checkpoint):
        super()._load_optimizer_and_scheduler(checkpoint)
        if checkpoint is None or self.exp_config.optimizer_config.optim != "adamw":
            return

        optimizer = getattr(self.optimizer, "optimizer", self.optimizer)
        if not isinstance(optimizer, torch.optim.AdamW):
            return
        for group in optimizer.param_groups:
            if not group.get("fused"):
                continue
            group["fused"] = False
            group["foreach"] = None
            if not group.get("capturable", False):
                for param in group["params"]:
                    step = optimizer.state.get(param, {}).get("step")
                    if isinstance(step, torch.Tensor):
                        optimizer.state[param]["step"] = step.cpu()
            logger.info("Resumed AdamW optimizer using the standard implementation")

    def _link_exp_config(self) -> TrainingArguments:
        """Build Hugging Face training arguments from the experiment config."""
        linked_args = {
            "output_dir": self.exp_config.trainer_config.output_dir,
            "max_steps": self.exp_config.trainer_config.num_train_steps,
            "per_device_train_batch_size": self.exp_config.trainer_config.per_device_train_batch_size,
            "gradient_accumulation_steps": self.exp_config.trainer_config.gradient_accumulation_steps,
            "save_strategy": self.exp_config.trainer_config.save_strategy,
            "save_steps": self.exp_config.trainer_config.save_steps,
            "save_total_limit": self.exp_config.trainer_config.save_total_limit,
            "save_only_model": self.exp_config.trainer_config.save_only_model,
            "logging_steps": self.exp_config.trainer_config.logging_steps,
            "dataloader_num_workers": self.exp_config.trainer_config.dataloader_num_workers,
            "dataloader_persistent_workers": getattr(
                self.exp_config.trainer_config, "dataloader_persistent_workers", False
            ),
            "dataloader_prefetch_factor": getattr(
                self.exp_config.trainer_config, "dataloader_prefetch_factor", None
            ),
            "dataloader_drop_last": True,
            # Keep model_max_length controlled by the model or processor config.
            "lr_scheduler_type": self.exp_config.trainer_config.lr_scheduler_type,
            "lr_scheduler_kwargs": self.exp_config.trainer_config.lr_scheduler_kwargs,
            "run_name": getattr(
                self.exp_config.trainer_config,
                "run_name",
                os.path.basename(self.exp_config.trainer_config.output_dir),
            ),
            "remove_unused_columns": False,
            "optim": "adamw_torch",
            "learning_rate": self.exp_config.optimizer_config.base_lr,
            "adam_beta1": self.exp_config.optimizer_config.adam_beta1,
            "adam_beta2": self.exp_config.optimizer_config.adam_beta2,
            "adam_epsilon": self.exp_config.optimizer_config.adam_epsilon,
            "warmup_steps": self.exp_config.optimizer_config.warmup_steps,
            "weight_decay": self.exp_config.optimizer_config.weight_decay,
            "seed": getattr(self.exp_config.trainer_config, "seed", 42),
            "data_seed": getattr(self.exp_config.trainer_config, "seed", 42),
        }
        if self.exp_config.model_config.precision_policy == FP32_MIXED_PRECISION_POLICY:
            linked_args.update(
                {
                    "bf16": True,
                    "fp16": False,
                    "tf32": TF32_ENABLED,
                }
            )
        else:
            linked_args.update(
                {
                    "bf16": self.exp_config.trainer_config.bf16,
                    "fp16": False,
                    "tf32": self.exp_config.trainer_config.tf32,
                }
            )
        linked_args["max_grad_norm"] = 1.0

        wandb_project = self.exp_config.trainer_config.wandb_project
        if wandb_project:
            os.environ["WANDB_PROJECT"] = wandb_project
            linked_args["report_to"] = ["wandb"]
            if os.environ.get("WANDB_DISABLED", "").lower() in {
                "true",
                "1",
                "yes",
                "on",
            }:
                os.environ["WANDB_DISABLED"] = "false"
        else:
            linked_args["report_to"] = []

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if self.exp_config.trainer_config.fsdp1 and world_size > 1:
            linked_args["fsdp"] = "shard_grad_op"
            linked_args["fsdp_config"] = {
                "backward_prefetch": "BACKWARD_PRE",
                "use_orig_params": True,
                "sync_module_states": True,
            }
            linked_args["ddp_find_unused_parameters"] = False
            linked_args["seed"] += int(os.environ.get("RANK", 0))
        else:
            linked_args["ddp_find_unused_parameters"] = True
        if linked_args["dataloader_num_workers"] == 0:
            linked_args["dataloader_prefetch_factor"] = None
            linked_args["dataloader_persistent_workers"] = False
        training_args = TrainingArguments(**linked_args)
        return training_args

    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        # Preserve the latest-microbatch metric semantics without synchronizing
        # the GPU between forward and backward. Reduce only when Trainer logs.
        self.loss_cache.update(
            (key, value.detach())
            for key, value in outputs.items()
            if key.endswith("_loss")
        )
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        cache = getattr(self, "loss_cache", {})
        if cache:
            names = sorted(cache)
            values = torch.stack([cache[name].float() for name in names])
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
                values /= torch.distributed.get_world_size()
            logs.update(zip(names, values.cpu().tolist(), strict=True))
        logs["step"] = self.state.global_step
        super().log(logs, start_time)


def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str) -> None:
    """Collect the model state dict on CPU and save it with the Trainer."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
