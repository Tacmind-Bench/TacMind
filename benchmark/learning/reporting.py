"""Save evaluation results and execution errors."""
from contextlib import contextmanager

from benchmark.io import write_json


@contextmanager
def evaluation_report(report, path):
    try:
        yield
        rows = report["episodes"]
        requested = report["episodes_requested"]
        if requested <= 0 or len(rows) != requested:
            raise ValueError(f"Incomplete evaluation: completed {len(rows)} of {requested} requested episodes")
        successes = sum(row["success"] for row in rows)
        report.update(status="completed", episodes_completed=len(rows), successes=successes,
                      success_rate=successes / len(rows), score_complete=True)
    except BaseException as error:
        report.pop("success_rate", None)
        report.update(status="error", score_complete=False,
                      error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(path, report)
