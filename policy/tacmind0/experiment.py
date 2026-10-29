"""Translate benchmark settings into the native VTLA experiment configuration."""

from pathlib import Path


def build_experiment(cfg, checkpoint=None, *, data=None, tactile=None):
    from tacmind0.policy.experiment import TacMind0Experiment

    exp = TacMind0Experiment()
    exp.model_config.model_name_or_path = str(checkpoint or cfg["pretrained_path"])
    exp.model_config.chunk_size = cfg["chunk_size"]
    if tactile is not None:
        exp.model_config.replace_tactile_encoder = True
        exp.model_config.tactile_weights = str(tactile / "encoder.pt")
        exp.model_config.backbone_config = str(tactile / "backbone_config.json")
    exp.data_config.jsonl_dir = str(data or Path(cfg["output_dir"]) / "tacmind0_data")
    exp.data_config.image_dir = cfg["prepared_root"]
    exp.data_config.marker_cache_size = cfg["model"].get("marker_cache_size", 256)
    exp.data_config.norm_file = str(
        Path(cfg["output_dir"]) / "normalization" / "norm_stats.json"
    )
    exp.inference_config.device = cfg.get("device", "cuda:0")
    exp.inference_config.diffusion_steps = cfg["model"]["diffusion_steps"]
    for setting in ("action_backend", "cuda_graphs"):
        if setting in cfg["model"]:
            setattr(exp.inference_config, setting, cfg["model"][setting])
    exp.optimizer_config.base_lr = cfg["learning_rate"]
    trainer = exp.trainer_config
    trainer.output_dir = str(Path(cfg["output_dir"]) / "policy")
    trainer.num_train_steps = cfg["steps"]
    trainer.seed = cfg["seed"]
    trainer.run_name = Path(cfg["output_dir"]).name
    trainer.per_device_train_batch_size = cfg["batch_size"]
    trainer.save_steps = cfg["save_freq"]
    trainer.dataloader_num_workers = cfg["num_workers"]
    trainer.dataloader_persistent_workers = cfg["num_workers"] > 0
    trainer.dataloader_prefetch_factor = 2 if cfg["num_workers"] else None
    trainer.fsdp1 = False
    return exp
