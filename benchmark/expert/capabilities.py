"""Backend and task combinations implemented by the expert collectors."""

from __future__ import annotations

from benchmark.catalog import TASK_IDS

BACKENDS = ("gsmini",)
SUITES = ("precision", "recog", "dynamic", "extra")
ENVIRONMENT_TASK_IDS = {"extra": {"gsmini": tuple(range(55))}}
REUSABLE_GSMINI_TASK_IDS = {
    "precision": (0, 1, 5), "recog": (0, 4, 5, 8, 9, 10), "dynamic": (4, 6),
}
SHARED_TABLETOP_EXPERT_FAMILIES = frozenset(("tube", "key", "hdmi", "gear"))


def supported_task_ids(suite: str, backend: str) -> tuple[int, ...]:
    if suite == "extra" and backend == "gsmini":
        from benchmark.catalog import is_extra_scene, list_tasks
        from benchmark.tasks.extra.gsmini_scenes import EXPERTS
        return tuple(task["task_id"] for task in list_tasks("tac_extra", resolve=False)
                     if not is_extra_scene(task)
                     or task.get("episode_config", {}).get("scene_template") == "extra8"
                     or task["source"]["scene"] in EXPERTS)
    return TASK_IDS[f"tac_{suite}"]


def process_reuse_supported(suite: str, task_id: int) -> bool:
    # Static scenes must retain their session: rebuilding them in the
    # same Kit process can hang after env.close(). Other Dynamic tasks replace
    # their stage between episodes; other suites need a fresh process per seed.
    return (task_id in REUSABLE_GSMINI_TASK_IDS.get(suite, ())
            or (suite == "dynamic" and task_id in TASK_IDS["tac_dynamic"]))
