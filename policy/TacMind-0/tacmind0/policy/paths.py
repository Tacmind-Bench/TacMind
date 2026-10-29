"""Default source-checkout resource locations; CLI paths can override these."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
