"""Build LeWM HDF5 mosaics from one or more gsmini LeRobot datasets."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import pandas as pd

from tacbenchhub.data.tactile_history_video import (
    DEFAULT_FPS,
    capped_episode_rows,
    decode_video,
    episode_slice,
    resize_history_mosaic,
    stack_rgb_marker_mosaic,
    tile_tactile_history,
)

PIXEL_MODES = ("rgb_marker", "marker")


def _video_path(dataset: Path, key: str, coords: dict) -> Path:
    return (
        dataset
        / "videos"
        / key
        / f"chunk-{int(coords['chunk']):03d}"
        / f"file-{int(coords['file']):03d}.mp4"
    )


def load_stream(
    cache: dict[Path, list[np.ndarray]],
    dataset: Path,
    key: str,
    coords: dict,
    length: int,
) -> list[np.ndarray]:
    path = _video_path(dataset, key, coords)
    if path not in cache:
        print(f"decode {key} {path.name}", flush=True)
        cache[path] = decode_video(path)
    return episode_slice(
        cache[path],
        from_timestamp=float(coords["from"]),
        to_timestamp=float(coords["to"]),
        length=int(length),
    )


def concat_lerobot_table(dataset: Path, columns: list[str]) -> pd.DataFrame:
    paths = sorted((dataset / "data").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet under {dataset / 'data'}")
    return pd.concat(
        [pd.read_parquet(path, columns=columns) for path in paths],
        ignore_index=True,
    )


def sorted_episode_groups(frame: pd.DataFrame) -> Iterator[tuple[int, pd.DataFrame]]:
    for episode, group in frame.groupby("episode_index"):
        yield int(episode), group.sort_values("frame_index", kind="mergesort")


def load_actions(dataset: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    table = concat_lerobot_table(
        dataset, ["episode_index", "frame_index", "action", "observation.state"]
    )
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for episode, ordered in sorted_episode_groups(table):
        out[episode] = (
            _stack_series(ordered, "action"),
            _stack_series(ordered, "observation.state"),
        )
    return out


def _stack_series(ordered: pd.DataFrame, column: str) -> np.ndarray:
    return np.stack(
        [np.asarray(value, dtype=np.float32) for value in ordered[column].to_numpy()]
    )


def load_vector_column(dataset: Path, column: str, *, width: int) -> dict[int, np.ndarray]:
    table = concat_lerobot_table(dataset, ["episode_index", "frame_index", column])
    out: dict[int, np.ndarray] = {}
    for episode, ordered in sorted_episode_groups(table):
        rows = [
            np.asarray(value, dtype=np.float32).reshape(int(width))
            for value in ordered[column].to_numpy()
        ]
        out[episode] = np.stack(rows, axis=0)
    return out


def load_marker_streams(
    cache: dict[Path, list[np.ndarray]],
    dataset: Path,
    row: dict[str, Any],
    length: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    episode = int(row["episode_index"])
    if "marker_left" not in row or "marker_right" not in row:
        raise KeyError(f"{dataset} episode {episode} is missing marker video")
    left = load_stream(
        cache, dataset, "observation.images.marker_left", row["marker_left"], length
    )
    right = load_stream(
        cache, dataset, "observation.images.marker_right", row["marker_right"], length
    )
    return left, right


def write_preview(path: Path, frames: list[np.ndarray], fps: float = DEFAULT_FPS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    scale = 0.5
    out_w = max(2, int(width * scale) // 2 * 2)
    out_h = max(2, int(height * scale) // 2 * 2)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_w, out_h),
    )
    for frame in frames:
        writer.write(cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA))
    writer.release()


def infer_task_id(dataset: Path, explicit: int | None = None) -> int:
    if explicit is not None:
        return int(explicit)
    name = dataset.name
    if name.startswith("task_"):
        suffix = name.split("_", 1)[1]
        if suffix.isdigit():
            return int(suffix)
    return -1


def cli_dataset_sources(
    datasets: list[Path],
    task_ids: list[int] | None = None,
) -> list[tuple[Path, int]]:
    """Pair ``--dataset`` paths with aligned ``--task-id`` values (or inferred ids)."""
    ids = list(task_ids or [])
    if ids and len(ids) != len(datasets):
        raise ValueError("--task-id count must match --dataset count")
    sources: list[tuple[Path, int]] = []
    for index, path in enumerate(datasets):
        resolved = Path(path).expanduser()
        explicit = ids[index] if ids else None
        sources.append((resolved, infer_task_id(resolved, explicit)))
    return sources


def _episode_payload(
    *,
    pixels: list[np.ndarray],
    action: np.ndarray,
    proprio: np.ndarray,
    task_id: int,
) -> dict[str, Any]:
    n_frames = len(pixels)
    return {
        "pixels": pixels,
        "action": action[:n_frames],
        "proprio": proprio[:n_frames],
        "task_id": np.full((n_frames,), int(task_id), dtype=np.int32),
    }


def build_sources(
    sources: list[tuple[Path, int]],
    h5_path: Path,
    *,
    pixels_mode: str,
    out_size: int,
    max_episodes: int | None,
    preview_episodes: int,
    preview_dir: Path,
) -> dict[str, int]:
    from stable_worldmodel.data.formats.hdf5 import HDF5Writer

    mode = str(pixels_mode)
    if mode not in PIXEL_MODES:
        raise ValueError(f"pixels_mode must be one of {PIXEL_MODES}, got {mode}")
    if not sources:
        raise ValueError("at least one LeRobot dataset is required")
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    written_episodes = 0
    written_frames = 0
    with HDF5Writer(h5_path, mode="overwrite") as writer:
        for dataset, task_id in sources:
            cache: dict[Path, list[np.ndarray]] = {}
            actions = load_actions(dataset)
            rows = capped_episode_rows(dataset, max_episodes)
            remaining_previews = int(preview_episodes)
            for row in rows:
                episode = int(row["episode_index"])
                length = int(row["length"])
                left_m, right_m = load_marker_streams(cache, dataset, row, length)
                n_frames = min(len(left_m), len(right_m))
                left_rgb = right_rgb = None
                if mode == "rgb_marker":
                    left_rgb = load_stream(
                        cache, dataset, "observation.images.tactile_left", row["tactile_left"], length
                    )
                    right_rgb = load_stream(
                        cache, dataset, "observation.images.tactile_right", row["tactile_right"], length
                    )
                    n_frames = min(n_frames, len(left_rgb), len(right_rgb))
                action, proprio = actions[episode]
                n_frames = min(n_frames, len(action), len(proprio))
                left_m = left_m[:n_frames]
                right_m = right_m[:n_frames]
                if mode == "rgb_marker":
                    assert left_rgb is not None and right_rgb is not None
                    left_rgb = left_rgb[:n_frames]
                    right_rgb = right_rgb[:n_frames]
                print(
                    f"task {task_id:02d} episode {episode:02d} frames={n_frames} pixels={mode}",
                    flush=True,
                )
                if mode == "marker":
                    pixels = [
                        resize_history_mosaic(left_m, right_m, index, out_size=out_size)
                        for index in range(n_frames)
                    ]
                    preview_full = [
                        tile_tactile_history(left_m, right_m, index) for index in range(n_frames)
                    ]
                else:
                    pixels = [
                        stack_rgb_marker_mosaic(
                            left_rgb, right_rgb, left_m, right_m, index, out_size=out_size
                        )
                        for index in range(n_frames)
                    ]
                    preview_full = [
                        np.concatenate(
                            [
                                tile_tactile_history(left_rgb, right_rgb, index),
                                tile_tactile_history(left_m, right_m, index),
                            ],
                            axis=0,
                        )
                        for index in range(n_frames)
                    ]
                writer.write_episode(
                    _episode_payload(
                        pixels=pixels,
                        action=action,
                        proprio=proprio,
                        task_id=int(task_id),
                    )
                )
                written_episodes += 1
                written_frames += n_frames
                if remaining_previews > 0:
                    preview_name = f"task_{int(task_id):02d}_episode_{episode:03d}.mp4"
                    write_preview(preview_dir / preview_name, preview_full)
                    remaining_previews -= 1
    with h5py.File(h5_path, "a") as handle:
        handle.attrs["n_episodes"] = int(written_episodes)
        handle.attrs["out_size"] = int(out_size)
        handle.attrs["pixels_mode"] = mode
        handle.attrs["mosaic_resize"] = "per_cell"
    print(
        f"wrote {h5_path} episodes={written_episodes} frames={written_frames}",
        flush=True,
    )
    return {"episodes": written_episodes, "frames": written_frames}
