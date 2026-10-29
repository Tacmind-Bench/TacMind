"""Git-tracked LeWM Hydra helpers: extra config dir, vendor trainer."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from tacbenchhub.models.lewm.dinov3 import DEFAULT_DINOV3_HUB

ROOT = Path(__file__).resolve().parents[3]
LEWM = ROOT / "third_party" / "le-wm"
CONFIG_DIR = ROOT / "configs" / "lewm"
VENDOR_CONFIG_DIR = LEWM / "config" / "train"
DEFAULT_STABLEWM_HOME = ROOT / "workspace" / "lewm"
DATA_CONFIG = "tacbench_gear_t36_marker"
VISUO_JEPA_DATA = "tacbench_multitask_visuo_jepa"
VISUO_JEPA_MODEL = "visuo_jepa"
VISUO_JEPA_BATCH_SIZE = "32"
VISUO_JEPA_DEVICES = "1"
VISUO_JEPA_NUM_WORKERS = "4"


def require_vendor_clone() -> Path:
    """Raise if ``third_party/le-wm`` is missing (gitignored local clone)."""
    marker = LEWM / "train.py"
    if not marker.is_file():
        raise FileNotFoundError(
            f"third_party/le-wm is missing ({LEWM}). Clone lucas-maes/le-wm "
            "there before running visuo-JEPA or vendor mosaic training."
        )
    return LEWM


def prepare_environ() -> None:
    os.environ.setdefault("STABLEWM_HOME", str(DEFAULT_STABLEWM_HOME))
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("DINOV3_PATH", DEFAULT_DINOV3_HUB)
    os.environ.setdefault("WANDB_MODE", "offline")
    # ``le-wm`` is a local checkout with top-level ``train``, ``module``, and
    # ``utils`` modules rather than an installable package. Keep this import
    # boundary explicit and local to the vendor launcher.
    for path in (str(LEWM), str(ROOT)):
        if path not in sys.path:
            sys.path[:0] = [path]


def hydra_argv(argv: list[str] | None = None) -> list[str]:
    """Point Hydra at vendor ``lewm.yaml`` plus first-party ``configs/lewm``."""
    args = list(sys.argv if argv is None else argv)
    extras: list[str] = []
    if "--config-path" not in args:
        extras.extend(["--config-path", str(VENDOR_CONFIG_DIR)])
    if "--config-dir" not in args:
        extras.extend(["--config-dir", str(CONFIG_DIR)])
    if extras:
        args[1:1] = extras
    return args


def _first_hydra_override_index(args: list[str]) -> int:
    """Index of the first Hydra override (after ``--config-*`` flags)."""
    index = 1
    while index < len(args) and args[index] in ("--config-path", "--config-dir"):
        index += 2
    return index


def visuo_jepa_argv(argv: list[str] | None = None) -> list[str]:
    """Vendor Hydra dirs plus visuo-JEPA model/data and safer loader defaults."""
    args = hydra_argv(argv)
    # Vendor lewm.yaml has no pred_mask. Insert +pred_mask=none *before* a
    # later pred_mask=contact/last override so Hydra struct compose succeeds.
    if not any(item.startswith("+pred_mask=") for item in args[1:]):
        args.insert(_first_hydra_override_index(args), "+pred_mask=none")
    overrides = [
        item.split("=", 1)[0].lstrip("+")
        for item in args[1:]
        if "=" in item and not item.startswith("--")
    ]
    defaults = {
        "model": f"model={VISUO_JEPA_MODEL}",
        "data": f"data={VISUO_JEPA_DATA}",
        "loader.batch_size": f"loader.batch_size={VISUO_JEPA_BATCH_SIZE}",
        "trainer.devices": f"trainer.devices={VISUO_JEPA_DEVICES}",
        "num_workers": f"num_workers={VISUO_JEPA_NUM_WORKERS}",
    }
    for key, item in defaults.items():
        if key not in overrides:
            args.append(item)
    return args


def run() -> None:
    require_vendor_clone()
    prepare_environ()
    sys.argv = hydra_argv(sys.argv)
    os.chdir(LEWM)
    import train as lewm_train

    lewm_train.run()
