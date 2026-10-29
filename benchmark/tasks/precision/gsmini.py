"""Precision collection through the shared tabletop expert strategies."""

from benchmark.catalog import get_task, tabletop_variant
from benchmark.expert.tabletop import run as collect_tabletop


def run(app, seed, output, task_id=0, *, session=None):
    task = get_task("tac_precision", task_id)
    return collect_tabletop(app, seed, output, tabletop_variant(task), session=session)
