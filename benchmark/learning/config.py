"""Dataset conversion and policy input configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from benchmark.catalog import ROOT


def policy_input_features(features):
    """Derive RGB camera and robot-state shapes from dataset metadata."""
    from benchmark.contracts import VISION_CAMERAS
    from lerobot.datasets.utils import dataset_to_policy_features

    keys = {"observation.state", *(f"observation.images.{name}" for name in VISION_CAMERAS)}
    return dataset_to_policy_features({key: value for key, value in features.items() if key in keys})


@dataclass(frozen=True)
class ConversionConfig:
    collection_manifest: Path
    dataset_root: Path
    repo_id: str
    suite: str = "tac_precision"
    backend: str = "gsmini"
    task_id: int = 0
    episodes: int = 100
    camera_names: tuple[str, ...] | None = ("agentview", "wristview")
    include_markers: bool = False
    include_deformation: bool = False
    video_gop: int = 8
    video_crf: int = 30
    video_preset: int = 8

    def __post_init__(self):
        if type(self.episodes) is not int or self.episodes <= 0:
            raise ValueError("Conversion episodes must be a positive integer")
        for name in ("collection_manifest", "dataset_root"):
            path = Path(getattr(self, name)).expanduser()
            object.__setattr__(self, name, path if path.is_absolute() else ROOT / path)
        if self.camera_names is not None:
            object.__setattr__(self, "camera_names", tuple(self.camera_names))

    @classmethod
    def from_file(cls, path: str | Path) -> ConversionConfig:
        config = json.loads(Path(path).read_text())
        return cls(**config)

    def to_dict(self) -> dict:
        result = asdict(self)
        for name in ("collection_manifest", "dataset_root"):
            path = getattr(self, name)
            result[name] = str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
        return result
