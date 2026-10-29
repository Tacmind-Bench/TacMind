"""Worker entrypoints for TacMind-0; implementations live in policy.tacmind0."""

from policy.tacmind0.inference import TacMind0Inference
from policy.tacmind0.training import train as train

__all__ = ["load", "train"]


def load(cfg, checkpoint):
    return TacMind0Inference(cfg, checkpoint)
