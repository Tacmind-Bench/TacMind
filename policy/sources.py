"""Locate bundled model sources without importing their runtime dependencies."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def checkout_path(name, *, root=ROOT):
    """Keep each model source directory alongside its adapter under policy/."""
    directory = "TacMind-0" if name == "tacmind0" else name
    return Path(root) / "policy" / directory
