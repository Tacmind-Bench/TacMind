"""Isolate the historical Hydra import paths from the downstream VTLA runtime."""
import sys
from pathlib import Path

root = Path(__file__).resolve().parent / "upstream"
vendor = root / "third_party/le-wm"
sys.path[:0] = [str(root), str(vendor), str(vendor / "stable-worldmodel")]

if __name__ == "__main__":
    from tacbenchhub.models.lewm.visuo_jepa_train import run
    run()
