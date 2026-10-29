"""BaseTrainerConfig: shared TacMind-0 runtime behavior."""

from dataclasses import dataclass, field


@dataclass
class BaseTrainerConfig:
    fsdp1: bool | None = True
    output_dir: str = "user_checkpoints/training"
    num_train_steps: int = 100000
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    save_strategy: str = "steps"
    save_steps: int = 5000
    save_total_limit: int = 50
    save_only_model: bool = False
    logging_steps: int = 1
    dataloader_num_workers: int = 4
    dataloader_persistent_workers: bool = True
    dataloader_prefetch_factor: int = 2
    model_max_length: int = 1024
    bf16: bool = True
    tf32: bool = True
    lr_scheduler_type: str = "cosine_with_min_lr"
    lr_scheduler_kwargs: dict = field(default_factory=lambda: {"min_lr_rate": 0.1})
    wandb_project: str | None = ""
