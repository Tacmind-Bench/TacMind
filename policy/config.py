"""Small, dependency-free configuration shared across isolated policy environments."""
import json
import os
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def worker_environment():
    """Keep Isaac's Python/library overlays out of isolated policy processes."""
    return {key: value for key, value in os.environ.items()
            if key not in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH")}


def resolve(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def interpreter_path(value):
    """Keep venv/uv interpreter symlinks: resolving them changes sys.prefix."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_config(path):
    return normalize_config(json.loads(Path(path).expanduser().read_text()))


def normalize_config(config, *, stage="train"):
    """Resolve paths and recipe defaults without mutating the caller's data."""
    cfg = deepcopy(config)
    if "dataset" not in cfg and stage == "train":
        manifest = json.loads((resolve(cfg["prepared_root"]) / "manifest.json").read_text())
        cfg["dataset"] = manifest["dataset"]
        cfg.setdefault("prompt", manifest.get("prompt"))
    cfg["dataset"] = dict(cfg["dataset"], root=str(resolve(cfg["dataset"]["root"])))
    cfg.setdefault("prepared_root", str(Path(cfg["dataset"]["root"]) / "policy_exports" / cfg["policy_type"]))
    cfg["prepared_root"] = str(resolve(cfg["prepared_root"]))
    if stage == "prepare":
        return cfg
    # The shipped templates are the single source of native recipe defaults.
    template = json.loads((ROOT / "policy/configs" / f"{cfg['policy_type']}.json").read_text())
    defaults = {key: template[key] for key in (
        "steps", "batch_size", "chunk_size", "save_freq", "num_workers", "learning_rate", "seed")}
    defaults.update(device="cuda:0", model={})
    defaults["n_action_steps"] = cfg.get("chunk_size", defaults["chunk_size"])
    if cfg["policy_type"] == "act_tactile":
        defaults.update(temporal_ensemble_coeff=template["temporal_ensemble_coeff"],
                        validation_fraction=template["validation_fraction"])
        if cfg.get("temporal_ensemble_coeff", defaults["temporal_ensemble_coeff"]) is not None:
            defaults["n_action_steps"] = 1
    cfg = {**defaults, **cfg}
    cfg["output_dir"] = str(resolve(cfg["output_dir"]))
    cfg["runtime_python"] = str(interpreter_path(cfg["runtime_python"]))
    if cfg.get("pretrained_path"):
        cfg["pretrained_path"] = str(resolve(cfg["pretrained_path"]))
    if cfg["policy_type"] == "act_tactile" and cfg["model"].get("tactile_ckpt"):
        cfg["model"]["tactile_ckpt"] = str(resolve(cfg["model"]["tactile_ckpt"]))
    if cfg["policy_type"] == "tacmind0":
        from policy.tacmind0.config import normalize_model
        cfg["model"] = normalize_model(cfg["model"])
    return cfg


def cameras(policy_type):
    return ("agentview", "wristview", "gsmini_left", "gsmini_right") if policy_type in (
        "act_tactile", "n0-vtla") else ("agentview", "wristview")


