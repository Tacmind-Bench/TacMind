"""Keep dataset metadata with training outputs for later evaluation."""

import json
from pathlib import Path

from benchmark.catalog import ROOT


def checkpoint_metadata(checkpoint):
    checkpoint = Path(checkpoint).expanduser().resolve()
    training = json.loads((checkpoint / "train_config.json").read_text())
    dataset = Path(training["dataset"]["root"]).expanduser()
    dataset = dataset if dataset.is_absolute() else ROOT / dataset
    current = (training.get("training_provenance") or {}).get("dataset")
    if current is not None:
        metadata = current["metadata"]
    else:
        path = checkpoint / "conversion_metadata.json"
        metadata = json.loads((path if path.is_file() else dataset / "conversion_metadata.json").read_text())
    return training, dataset, metadata


def dataset_provenance(root, episodes=None):
    root = Path(root)
    metadata = json.loads((root / "conversion_metadata.json").read_text())
    info = json.loads((root / "meta/info.json").read_text())
    selected = list(range(info["total_episodes"])) if episodes is None else list(episodes)
    seeds = metadata.get("seeds", [])
    return {"metadata": metadata, "episodes": selected,
            "seeds": [seeds[index] for index in selected] if seeds else []}


def training_provenance(root, episodes=None):
    return {"dataset": dataset_provenance(root, episodes)}
