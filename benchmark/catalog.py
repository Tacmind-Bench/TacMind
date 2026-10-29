"""Simulator-independent access to the Tac task suites."""

from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
SUITES = Path(__file__).resolve().parent / "tasks"
TASK_IDS = {
    "tac_precision": tuple(range(6)),
    "tac_recog": tuple(range(11)),
    "tac_dynamic": tuple(range(11)),
    "tac_extra": tuple(range(100)),
}
COUNTS = {suite: len(ids) for suite, ids in TASK_IDS.items()}
ROBOT_ASSET = Path('robots/Franka_gsmini/tacmind.usda')


@lru_cache(maxsize=4)
def _read_suite(name: str) -> dict:
    """Cache task definitions internally."""
    if name not in COUNTS:
        raise ValueError(f"Unknown suite {name!r}; expected one of {tuple(COUNTS)}")
    data = json.loads((SUITES / name.removeprefix("tac_") / "tasks.json").read_text(encoding="utf-8"))
    return data


# Desktop controller variants are internal strategy identifiers, independent of
# the public task numbering. Derive dispatch from the current catalog only.
TABLETOP_IDENTITIES = {
    task["episode_config"]["expert_variant"]: (suite, task["task_id"])
    for suite in ("tac_precision", "tac_recog", "tac_extra")
    for task in _read_suite(suite)["tasks"]
    if "expert_variant" in task.get("episode_config", {})
}


@lru_cache(maxsize=sum(COUNTS.values()))
def _read_task(name: str, task_id: int) -> dict:
    """Resolve assets only for the selected task and its shared template."""
    task = deepcopy(next(task for task in _read_suite(name)["tasks"] if task["task_id"] == task_id))
    if name == "tac_extra":
        if task.get("episode_config", {}).get("scene_template") == "extra8":
            from benchmark.tasks.extra.shared_cutting import resolve_task
            return resolve_task(task, _read_task(name, 8))
        if not is_extra_scene(task) and not is_tabletop_task(task):
            from benchmark.tasks.extra.visuals import configure_extra_presentation
            configure_extra_presentation(task)
    return task


def load_suite(name: str) -> dict:
    """Return an independent official suite with its original physics fields."""
    data = deepcopy(_read_suite(name))
    data["tasks"] = list_tasks(name)
    return data


def is_extra_scene(task: dict) -> bool:
    """Identify source-backed scenes within the single Extra catalog."""
    return task.get("reset_contract", {}).get("mode") == "extra_scene"


def is_tabletop_task(task: dict) -> bool:
    """Select the shared desktop strategies independently of the public suite."""
    return task.get("episode_config", {}).get("expert_family") in {
        "tube", "key", "hdmi", "gear", "tactile_reorientation", "tactile_search",
        "tactile_sorting", "tactile_insertion", "tactile_threading"}


def is_recognition_task(task: dict) -> bool:
    """Select search/placement goals; Recog also contains other strategy families."""
    return task.get("episode_config", {}).get("expert_family") in {"blind_search", "blind_migrated"}


def candidate_names(task: dict):
    """Enumerate search candidates independently of the requested target."""
    contract = task["reset_contract"]
    if contract["mode"] == "blind_permutation":
        names = tuple(role["object"] for role in contract["roles"].values())
    elif contract["mode"] == "ball_sorting":
        names = tuple(contract[key] for key in ("golf", "table_tennis", "blank"))
    else:
        names = tuple(name for group in contract["exchange_groups"] for name in group)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("Search requires distinct candidate objects")
    return names


def tabletop_variant(task: dict) -> int:
    """Keep strategy selection stable when public suite IDs are reassigned."""
    return task.get("episode_config", {}).get("expert_variant", task.get("task_id"))


def list_tasks(name: str | None = None, *, resolve: bool = True) -> list[dict]:
    """List released tasks in stable order; resolve=False reads only catalog metadata."""
    return deepcopy([_read_task(suite, task["task_id"]) if resolve else task
                     for suite in ((name,) if name else COUNTS)
                     for task in _read_suite(suite)["tasks"]])


def get_task(name: str, task_id: int) -> dict:
    if type(task_id) is not int or task_id not in TASK_IDS.get(name, ()):
        raise ValueError(f"Invalid task {name}:{task_id}")
    return deepcopy(_read_task(name, task_id))


def object_asset(asset_type: str) -> Path:
    """Resolve the local entry layer, falling back to the original source USD."""
    if asset_type.startswith("shared_cutting_"):
        from benchmark.tasks.extra.shared_cutting import food_asset
        return food_asset(asset_type)
    if asset_type in {"pc_ram_module", "pc_ram_socket"}:
        source = ASSETS / "objects" / asset_type / f"{asset_type.removeprefix('pc_')}.usda"
        if not source.is_file():
            raise FileNotFoundError(source)
        return source
    directory = ASSETS / "objects" / asset_type
    preferred = directory / "asset.usda"
    source = directory / f"{asset_type}.usd"
    if not source.is_file():
        raise FileNotFoundError(source)
    return preferred if preferred.is_file() else source


def scene_asset(task: dict) -> Path | None:
    layout = task.get("scene_layout")
    if not layout:
        return None
    path = ASSETS / "scenes" / layout
    if not path.is_file():
        raise FileNotFoundError(path)
    return path
