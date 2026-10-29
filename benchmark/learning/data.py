"""Recording adapters and the common official LeRobot dataset writer."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from benchmark.catalog import get_task
from benchmark.contracts import ACTION_REPRESENTATION, ObservationActionSpec, recording_contract
from benchmark.io.image_storage import rgb_sequence

from benchmark.learning.config import ConversionConfig
from benchmark.io import write_json


def _frame_image_statistics(frame, video_keys):
    """Compute safe moments using the native encoder's pixel conversion and sampling."""
    from lerobot.datasets.compute_stats import auto_downsample_height_width

    result = {}
    for key in video_keys:
        image = np.asarray(frame[key])
        if image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        image = auto_downsample_height_width(image.transpose(2, 0, 1)).astype(np.float64) / 255
        result[key] = {"min": image.min((1, 2), keepdims=True),
                       "max": image.max((1, 2), keepdims=True),
                       "mean": image.mean((1, 2), keepdims=True),
                       "std": image.std((1, 2), keepdims=True),
                       "count": np.array([image.shape[1] * image.shape[2]])}
    return result


def _write_image_standard_deviations(dataset, episodes):
    """Repair staged metadata after encoding; never replace a LeRobot function.

    LeRobot 0.4.4 squares uint8 pixels before casting, losing within-episode
    variance. Keep its videos and other statistics, and replace only image std.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.utils import write_stats

    for path in dataset.root.glob("meta/episodes/*/*.parquet"):
        table = pq.read_table(path)
        indices = table["episode_index"].to_pylist()
        for key in dataset.meta.video_keys:
            column = f"stats/{key}/std"
            field = table.schema.field(column)
            values = [episodes[index][key]["std"].tolist() for index in indices]
            table = table.set_column(table.schema.get_field_index(column), field, pa.array(values, type=field.type))
        pq.write_table(table, path)
    for key, statistics in aggregate_stats(episodes).items():
        dataset.meta.stats[key]["std"] = statistics["std"]
    write_stats(dataset.meta.stats, dataset.root)


Frame = dict[str, Any]
EpisodeFrames = Iterable[Frame]


def convert_episodes_to_lerobot(
    episodes: Iterable[EpisodeFrames], output_root: str | Path, *, repo_id: str,
    fps: int, robot_type: str, features: dict, encoder_threads: int = 4,
    conversion_metadata: dict | None = None,
    video_gop: int = ConversionConfig.video_gop,
    video_crf: int = ConversionConfig.video_crf,
    video_preset: int = ConversionConfig.video_preset,
) -> int:
    """Write MP4 frame streams through native LeRobot; publish only on success."""
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    output_root = Path(output_root)
    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        raise FileExistsError(f"LeRobot output must be empty: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=f".{output_root.name}-", dir=output_root.parent,
    ) as temporary:
        staged = Path(temporary) / "dataset"
        dataset = LeRobotDataset.create(
            repo_id=repo_id, root=staged, fps=fps, robot_type=robot_type,
            features=features, use_videos=True, video_backend="pyav",
            vcodec="libsvtav1",
            streaming_encoding=True, encoder_threads=encoder_threads,
            # Offline conversion must retain every frame. Native save_episode
            # drains each episode's queue before reading the next episode.
            encoder_queue_maxsize=0,
        )
        try:
            if dataset.meta.video_keys:
                # LeRobot 0.4.4 creates this native encoder before its first
                # episode; its public settings are copied to each camera thread.
                encoder = dataset._streaming_encoder
                encoder.g, encoder.crf, encoder.preset = video_gop, video_crf, video_preset
            image_statistics = []
            for episode in episodes:
                episode_statistics = {}
                for frame in episode:
                    dataset.add_frame(frame)
                    if dataset.meta.video_keys:
                        frame_statistics = _frame_image_statistics(frame, dataset.meta.video_keys)
                        episode_statistics = (aggregate_stats([episode_statistics, frame_statistics])
                                              if episode_statistics else frame_statistics)
                dataset.save_episode(parallel_encoding=False)
                image_statistics.append(episode_statistics)
        finally:
            dataset.finalize()
        if dataset.meta.video_keys:
            _write_image_standard_deviations(dataset, image_statistics)
        total_frames = dataset.num_frames
        if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
            raise FileExistsError(f"LeRobot output must be empty: {output_root}")
        if conversion_metadata is not None:
            write_json(staged / "conversion_metadata.json", {**conversion_metadata, "frames": total_frames})
        staged.rename(output_root)
    return total_frames


@dataclass
class HDF5EpisodeSource:
    """Adapt one task's successful pre-action recordings to the common frame stream."""
    manifest_path: Path
    suite: str
    task_id: int
    episodes: int
    backend: str = "gsmini"
    camera_names: tuple[str, ...] | None = None
    include_markers: bool = False
    include_deformation: bool = False

    def __post_init__(self):
        if type(self.episodes) is not int or self.episodes <= 0:
            raise ValueError("Conversion episodes must be a positive integer")
        if self.suite == "extra":
            self.suite = "tac_extra"
        self.manifest_path = Path(self.manifest_path)
        manifest = json.loads(self.manifest_path.read_text())
        if "episodes" in manifest:
            entries = manifest["episodes"].get(str(self.task_id), [])
        else:
            # Consume the existing collector's summary directly; no second
            # manifest-building script or copied episode list is required.
            entries = [{"file": job["output"], "seed": job["seed"], "frames": job["frames"]}
                       for job in manifest["jobs"] if job["status"] == "success"
                       and job["suite"].removeprefix("tac_") == self.suite.removeprefix("tac_")
                       and job["task_id"] == self.task_id]
        if len(entries) < self.episodes:
            raise ValueError(f"Requested {self.episodes} successful episodes for {self.suite}:{self.task_id}, "
                             f"but only {len(entries)} are available in {self.manifest_path}")
        self.entries = entries[:self.episodes]
        import h5py
        import hdf5plugin  # Register recording filters at the data I/O boundary.

        with h5py.File(self.manifest_path.parent / self.entries[0]["file"], "r") as recording:
            env_args = json.loads(recording["data"].attrs["env_args"])
        self.recording_spec = ObservationActionSpec.from_env_args(env_args)
        camera_names = self.recording_spec.camera_names if self.camera_names is None else tuple(self.camera_names)
        self.spec = replace(self.recording_spec, camera_names=camera_names)
        self.recording_contract = recording_contract(env_args)
        storage = env_args.get("storage", {})
        self.recording_storage = storage
        native_tiles = isinstance(storage, dict) and storage.get("rgb_chunk_shape") == [8, 60, 60, 3]
        batch_bytes = 8 * max(np.prod(self.spec.camera_shape(camera)) for camera in self.spec.camera_names)
        # Aligned native batches read tiles once; keep caching for other layouts and larger images.
        self._image_cache_bytes = 0 if native_tiles and batch_bytes <= 16 * 1024**2 else 64 * 1024**2
        self.instruction = get_task(self.suite, self.task_id)["language_instruction"]

    @contextmanager
    def _episode(self, entry):
        import h5py
        import hdf5plugin  # Register recording filters at the data I/O boundary.

        path = self.manifest_path.parent / entry["file"]
        with h5py.File(path, "r", rdcc_nbytes=self._image_cache_bytes) as recording:
            yield recording["data/demo_0"]

    def frames(self, entry: dict) -> Iterator[Frame]:
        with self._episode(entry) as demo, ThreadPoolExecutor(max_workers=len(self.spec.camera_names)) as pool:
            observations, length = demo["obs"], len(demo["actions"])
            episode_spec = ObservationActionSpec.from_env_args(json.loads(demo.parent.attrs["env_args"]))
            state_indices = [episode_spec.state_names.index(name) for name in self.spec.state_names]
            action_indices = [episode_spec.arm_joint_names.index(name) for name in self.spec.arm_joint_names]
            action_indices.append(-1)
            cameras = {camera: rgb_sequence(observations[camera]) for camera in self.spec.camera_names}
            tactile_images = {}
            if self.include_markers or self.include_deformation:
                from benchmark.backends.gsmini.markers import deformation_image, marker_image
                for side in ("left", "right"):
                    name = f"gsmini_{side}"
                    tactile_images[side] = cameras.get(name)
                    if tactile_images[side] is None:
                        tactile_images[side] = rgb_sequence(observations[name])
            # Use up to eight frames within a 16 MiB budget, or one larger frame.
            image_batch_size = max(1, min(8, (16 * 1024**2) // max(
                np.prod(images.shape[1:]) * images.dtype.itemsize for images in cameras.values())))
            for start in range(0, length, 64):
                states = np.asarray(observations["joint_pos"][start:start + 64], np.float32)[:, state_indices]
                targets = episode_spec.target_action(demo["actions"][start:start + 64])[:, action_indices]
                for offset in range(0, len(states), image_batch_size):
                    stop = min(offset + image_batch_size, len(states))
                    image_batch = dict(zip(cameras, pool.map(
                        lambda images: images[start + offset:start + stop], cameras.values()), strict=True))
                    for image_offset, (state, action) in enumerate(zip(states[offset:stop], targets[offset:stop])):
                        frame = {
                            "observation.state": state,
                            "action": action,
                            "task": self.instruction,
                        }
                        for camera in cameras:
                            frame[f"observation.images.{camera}"] = np.ascontiguousarray(
                                image_batch[camera][image_offset])
                        for side, images in tactile_images.items():
                            index = start + offset + image_offset
                            name = f"gsmini_{side}"
                            rgb = (image_batch[name][image_offset] if name in image_batch else images[index])
                            motion = observations[f"{name}_marker_motion"][index]
                            if self.include_markers:
                                frame[f"observation.images.marker_{side}"] = marker_image(rgb, motion)
                            if self.include_deformation:
                                frame[f"observation.images.deformation_{side}"] = deformation_image(rgb, motion)
                        yield frame

    def iter_episodes(self) -> Iterator[EpisodeFrames]:
        for index, entry in enumerate(self.entries):
            yield self.frames(entry)
            print(f"[CONVERT] {index + 1}/{self.episodes} {entry['file']} ({entry['frames']} frames)", flush=True)


def convert_dataset(config: ConversionConfig) -> int:
    source = HDF5EpisodeSource(
        manifest_path=config.collection_manifest, suite=config.suite, task_id=config.task_id,
        episodes=config.episodes, backend=config.backend, camera_names=config.camera_names,
        include_markers=config.include_markers,
        include_deformation=config.include_deformation,
    )
    metadata = {
        "source_manifest": str(config.collection_manifest), "suite": config.suite,
        "task_id": config.task_id, "backend": config.backend,
        "episodes": source.episodes,
        "seeds": [int(entry["seed"]) for entry in source.entries],
        "contract": source.spec.to_dict(),
        "recording_contract": source.recording_contract,
        "recording_storage": source.recording_storage,
        "action": ACTION_REPRESENTATION,
        "images": f"native LeRobot MP4 RGB videos; cameras: {list(source.spec.camera_names)}",
        "video_encoding": dict(encoder="libsvtav1", crf=config.video_crf,
                               gop=config.video_gop, preset=config.video_preset,
                               pixel_format="yuv420p", lossless=False),
    }
    features = source.spec.features()
    for view, enabled in (("marker", config.include_markers), ("deformation", config.include_deformation)):
        if enabled:
            for side in ("left", "right"):
                features[f"observation.images.{view}_{side}"] = {
                    "dtype": "image", "shape": source.recording_spec.camera_shape(f"gsmini_{side}"),
                    "names": ["height", "width", "channel"],
                }
    if config.include_markers:
        metadata["markers"] = "lossless RGB FOTS marker frames"
    if config.include_deformation:
        metadata["tactile_input"] = "fots-deformation-arrows-v1"
    return convert_episodes_to_lerobot(
        source.iter_episodes(), config.dataset_root, repo_id=config.repo_id,
        fps=source.spec.fps, robot_type=f"franka_{config.backend}", features=features,
        conversion_metadata=metadata,
        video_gop=config.video_gop, video_crf=config.video_crf, video_preset=config.video_preset,
    )
