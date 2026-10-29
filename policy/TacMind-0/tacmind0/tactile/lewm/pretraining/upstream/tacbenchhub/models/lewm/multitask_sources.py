"""Complete gsmini representative tasks for multi-task LeWM.

Skip incomplete collects (task 48, task 51). Mix ids use 50 episodes from
``tac_base_representatives_force_20260823`` (100 eps on disk; builders cap).
"""

from __future__ import annotations

from pathlib import Path

REPRESENTATIVES_REL = Path(
    "workspace/data/lerobot/gsmini/tac_base_representatives_force_20260823"
)
# 50-episode complete tasks only. Mix insertion families with search/sorting.
# The 20260823 dumps have 100 eps each; visuo mix builds cap at 50.
MULTITASK_TASK_IDS: tuple[int, ...] = (0, 12, 24, 36, 49, 50, 52)
TASK_NAMES: dict[int, str] = {
    0: "tube_target_material_select_3",
    12: "key_target_material_select_3",
    24: "hdmi_target_material_select_3",
    36: "gear_target_material_select_3",
    49: "tactile_object_search",
    50: "tactile_ball_sorting",
    52: "tactile_bulb_insertion",
}
H5_NAME = "tacbench_multitask_marker.h5"
OUTPUT_MODEL_NAME = "lewm_multitask_marker_fs1"
HYDRA_DATA = "tacbench_multitask_marker"


def dataset_roots(workspace: Path) -> list[tuple[int, Path]]:
    """Return ``(task_id, LeRobot root)`` for the multi-task mix."""
    base = workspace / REPRESENTATIVES_REL
    return [(int(task_id), base / f"task_{int(task_id)}") for task_id in MULTITASK_TASK_IDS]
