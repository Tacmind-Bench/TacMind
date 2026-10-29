"""Run the imported Tac-LeWM pretrainer against a local LeRobot dataset."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess

from policy.config import (
    ROOT,
    interpreter_path,
    resolve,
    worker_environment,
)
from policy.sources import checkout_path
from policy.checkpoint import atomic_checkpoint

SOURCE = checkout_path("tacmind0") / "tacmind0/tactile/lewm/pretraining"


def read_config(path):
    """Apply launcher defaults and resolve paths on the execution host."""
    template = json.loads((Path(__file__).parent / "configs/taclewm_pretrain.json").read_text())
    cfg = {key: template[key] for key in ("epochs", "devices", "batch_size", "num_workers", "seed")}
    cfg.update(json.loads(Path(path).expanduser().read_text()))
    for key in ("runtime_python", "dataset_path", "output_dir", "backbone_dir", "vision_backbone_dir"):
        cfg[key] = str(
            interpreter_path(cfg[key]) if key == "runtime_python" else resolve(cfg[key])
        )
    return cfg


def command(cfg):
    """Apply host settings to the Tac-LeWM pretraining recipe without accessing its data."""
    upstream = SOURCE / "upstream"
    overrides = [
        line[2:]
        for line in (upstream / "reference/pretraining_overrides.yaml").read_text().splitlines()
        if line.startswith("- ")
    ]
    replaced = {
        "data.dataset.name": json.dumps(cfg["dataset_path"]),
        "output_model_name": json.dumps(str(Path(cfg["output_dir"]) / "checkpoints")),
        "hydra.run.dir": json.dumps(str(Path(cfg["output_dir"]) / "hydra")),
        "model.encoder.model_id": json.dumps(cfg["backbone_dir"]),
        "model.encoder.cache_dir": json.dumps(
            str(Path(cfg["output_dir"]) / "model_cache")
        ),
        "model.dinov3_path": json.dumps(cfg["vision_backbone_dir"]),
        "model.allow_hf_download": "false",
        "trainer.max_epochs": str(cfg["epochs"]),
        "trainer.devices": str(cfg["devices"]),
        "loader.batch_size": str(cfg["batch_size"]),
        "loader.num_workers": str(cfg["num_workers"]),
        "num_workers": str(cfg["num_workers"]),
        "seed": str(cfg["seed"]),
        "loader.persistent_workers": str(cfg["num_workers"] > 0).lower(),
        "loader.prefetch_factor": "3" if cfg["num_workers"] else "null",
    }
    overrides = [
        arg for arg in overrides if arg.split("=", 1)[0].lstrip("+") not in replaced
    ]
    overrides += [f"{key}={value}" for key, value in replaced.items()]
    return [cfg["runtime_python"], str(ROOT / "policy/taclewm_lerobot_bootstrap.py"), *overrides]


def export_resources(cfg):
    """Export the model configuration files consumed by downstream fine-tuning."""
    from policy.tacmind0.representation import validate, write_contract

    output = Path(cfg["output_dir"])
    record = output / "run_config.json"
    validate(json.loads(record.read_text()) if record.is_file() else {}, source=record)
    checkpoints = output / "checkpoints"
    resources = output / "resources"
    sources = (
        (checkpoints / "config.json", "world_model_config.json"),
        (Path(cfg["backbone_dir"]) / "config.json", "backbone_config.json"),
        (
            Path(cfg["vision_backbone_dir"]) / "config.json",
            "vision_backbone_config.json",
        ),
    )
    if resources.exists() or resources.is_symlink():
        raise FileExistsError(resources)
    with atomic_checkpoint(resources) as staged:
        for src, name in sources:
            shutil.copy2(src, staged / name)
        write_contract(staged)
    write_contract(checkpoints)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Export resources from a completed run",
    )
    args = parser.parse_args(argv)
    cfg = read_config(args.config)
    if args.dry_run:
        print(
            json.dumps({"command": command(cfg), "dataset_download": False}, indent=2)
        )
        return
    if args.export_only:
        export_resources(cfg)
        return
    from policy.tacmind0.representation import TACTILE_INPUT, require_dataset

    require_dataset(cfg["dataset_path"])
    cfg["tactile_input"] = TACTILE_INPUT
    output = Path(cfg["output_dir"])
    if output.is_symlink() or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
        raise FileExistsError(f"Pretraining requires an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text(json.dumps(cfg, indent=2))
    env = worker_environment()
    # An unrelated server run must not turn this new training job into a resume.
    for key in list(env):
        if key.startswith("TACBENCH_INIT_") or key == "TACBENCH_RESUME_CKPT":
            env.pop(key)
    env.update(
        STABLEWM_HOME=str(output / "cache"),
        HF_HUB_OFFLINE="1",
        WANDB_MODE="offline",
        HF_DATASETS_OFFLINE="1",
        DINOV3_PATH=cfg["vision_backbone_dir"],
    )
    subprocess.run(command(cfg), cwd=ROOT, env=env, check=True)
    export_resources(cfg)


if __name__ == "__main__":
    main()
