"""Require the Deformation representation at TacMind-0 data and weight boundaries."""
import json
from pathlib import Path


TACTILE_INPUT = "fots-deformation-arrows-v1"
VIEWS = ("deformation_left", "deformation_right")


def validate(record, source):
    if record.get("tactile_input") != TACTILE_INPUT:
        raise ValueError(
            f"TacMind-0 requires {TACTILE_INPUT}: {source}. "
            "Rebuild data from the original HDF5 with include_deformation=true and use "
            "weights trained on Deformation; marker-only or unverified artifacts are incompatible."
        )


def require_dataset(root):
    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text())
    missing = [name for name in VIEWS if f"observation.images.{name}" not in info["features"]]
    if missing:
        raise ValueError(f"Missing Deformation images {missing}: {root}. "
                         "Convert the original HDF5 with include_deformation=true into a new dataset.")
    records = [path for path in (root / "conversion_metadata.json", root / "tacmind_lineage.json")
               if path.is_file()]
    if not records:
        validate({}, root)
    for path in records:
        validate(json.loads(path.read_text()), path)


def require_weights(path):
    path = Path(path)
    contract = (path if path.is_dir() else path.parent) / "tactile_input.json"
    validate(json.loads(contract.read_text()) if contract.is_file() else {}, contract)


def write_contract(directory):
    path = Path(directory) / "tactile_input.json"
    if path.is_file():
        validate(json.loads(path.read_text()), path)
    path.write_text(json.dumps({"tactile_input": TACTILE_INPUT}, indent=2) + "\n")
