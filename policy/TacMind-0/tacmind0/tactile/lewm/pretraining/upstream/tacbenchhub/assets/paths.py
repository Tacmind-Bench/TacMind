"""Canonical paths for bundled LIBERO configuration and external USD assets."""

from __future__ import annotations

import os
from pathlib import Path

from .resource_manifest import get_assets_path


def project_root() -> Path:
    """Return the checkout root containing ``tacbenchhub/``."""
    return Path(__file__).resolve().parents[2]


def task_config_dir() -> Path:
    """Return the packaged configuration directory for maintained suites."""
    override = os.environ.get("TACBENCH_TASK_CONFIG_DIR")
    if override:
        candidate = Path(override).expanduser().resolve()
        # Do not let a stale override from a different suite family hide the
        # packaged maintained configs.  A valid override must expose at least
        # one maintained-suite file.
        if any(candidate.glob("tac_*.json")):
            return candidate
    return project_root() / "assets" / "tasks" / "tac" / "config"


def config_dir_for_suite(suite: str) -> Path:
    """Resolve the configuration root selected by a suite identifier."""
    suite_name = str(suite)
    if suite_name.startswith("tac_"):
        override = os.environ.get("TACBENCH_TASK_CONFIG_DIR")
        candidate = Path(override).expanduser().resolve() if override else None
        if candidate is not None and (candidate / f"{suite_name}.json").is_file():
            return candidate
        return project_root() / "assets" / "tasks" / "tac" / "config"
    override = os.environ.get("LIBERO_CONFIG_DIR")
    candidate = Path(override).expanduser().resolve() if override else None
    if candidate is not None and (candidate / f"{suite_name}.json").is_file():
        return candidate
    return project_root() / "assets" / "tasks" / "libero" / "config"


def libero_config_dir() -> Path:
    """Return the internal LIBERO JSON configuration directory."""
    override = os.environ.get("LIBERO_CONFIG_DIR")
    if override:
        candidate = Path(override).expanduser().resolve()
        # A suite switch can leave the maintained root in this compatibility
        # variable.  Only accept an override that actually contains a legacy
        # suite file; otherwise use the packaged compatibility root.
        if any(candidate.glob("libero_*.json")):
            return candidate
    return project_root() / "assets" / "tasks" / "libero" / "config"


def libero_assets_dir() -> Path:
    """Return the repository-managed LIBERO object asset directory."""
    override = os.environ.get("LIBERO_ASSETS_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return get_assets_path() / "objects"


def assets_dir_for_suite(suite: str) -> Path:
    """Resolve object assets without carrying an override across suite families."""
    if str(suite).startswith("tac_"):
        override = os.environ.get("TACBENCH_ASSETS_DATA_DIR")
        if override:
            return Path(override).expanduser().resolve()
        return get_assets_path() / "objects"
    return libero_assets_dir()


def backgrounds_root() -> Path:
    """Return the bundled HDR/EXR environment-map directory."""
    override = os.environ.get("TACBENCHHUB_BACKGROUNDS_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return get_assets_path() / "background"


def belt_assets_root() -> Path:
    """Return the managed root containing task scene layers."""
    override = os.environ.get("TACBENCHHUB_BELT_ASSETS_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return get_assets_path() / "scenes"
