"""Validated marker episodes and train-only action statistics for both stages."""

import json
import random
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=4)
def read_episode_rows(path: str) -> list[dict]:
    rows = [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]
    if len(rows) < 2:
        raise ValueError(f"Need at least two frames: {path}")
    for row in rows:
        action = np.asarray(row["action"], dtype=np.float32)
        if action.shape != (8,) or not np.isfinite(action).all():
            raise ValueError(f"Expected finite raw 8D action in {path}")
        for key in ("marker_left", "marker_right", "images_1"):
            if key not in row:
                raise ValueError(f"Missing {key} in {path}")
    return rows


def split_episodes(
    root: Path, fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must lie in (0,1)")
    files = sorted(str(path.resolve()) for path in root.rglob("*.jsonl"))
    if len(files) < 2:
        raise ValueError("Need at least two episodes for disjoint train/validation")
    random.Random(seed).shuffle(files)
    count = max(1, min(len(files) - 1, round(len(files) * fraction)))
    return files[count:], files[:count]


def compute_action_statistics(files: list[str]) -> dict:
    total, squares, count = np.zeros(8), np.zeros(8), 0
    for file in files:
        actions = np.asarray(
            [row["action"] for row in read_episode_rows(file)], dtype=np.float64
        )
        total += actions.sum(0)
        squares += (actions * actions).sum(0)
        count += len(actions)
    mean = total / count
    std = np.sqrt(np.maximum(squares / count - mean * mean, 0)).clip(1e-6)
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "count": count,
        "representation": "raw JSONL absolute action; train-only per-channel zscore",
    }
