#!/usr/bin/env python3
"""Export aligned LeWM HDF5 markers to episode JSONL and lossless FFV1 videos.

Current 42x42 marker frames are exported at 20 Hz for stride-5 history sampling.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np


FPS = 20
FRAME_SIZE = 42
VIDEO_FILES_PER_CHUNK = 100


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _new_encoder(path: Path) -> subprocess.Popen:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{FRAME_SIZE}x{FRAME_SIZE}",
        "-framerate",
        str(FPS),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-pixel_format",
        "bgr0",
        "-g",
        "1",
        str(path),
    ]
    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def _finish_encoder(process: subprocess.Popen) -> None:
    if process.stdin is not None:
        process.stdin.close()
    stderr = process.stderr.read() if process.stderr is not None else b""
    status = process.wait()
    if status:
        raise RuntimeError(
            f"FFV1 encoder exited with status {status}: {stderr.decode(errors='replace')}"
        )


def _encode_marker_pair(
    pixels: h5py.Dataset,
    global_start: int,
    global_end: int,
    left_path: Path,
    right_path: Path,
    *,
    read_batch: int = 128,
) -> None:
    left = _new_encoder(left_path)
    right = _new_encoder(right_path)
    try:
        for start in range(global_start, global_end, read_batch):
            stop = min(start + read_batch, global_end)
            # Dataset layout: (rows, finger, history, BGR, height, width).
            block = pixels[start:stop, :, -1]
            left_frames = np.ascontiguousarray(block[:, 0].transpose(0, 2, 3, 1))
            right_frames = np.ascontiguousarray(block[:, 1].transpose(0, 2, 3, 1))
            left.stdin.write(left_frames.tobytes())
            right.stdin.write(right_frames.tobytes())
    except Exception:
        for process in (left, right):
            process.terminate()
            process.wait()
        raise
    _finish_encoder(left)
    _finish_encoder(right)


def _absolute_camera_url(
    row: dict[str, Any], field: str, source_image_root: Path
) -> None:
    entry = row[field]
    url = Path(entry["url"])
    if not url.is_absolute():
        url = source_image_root / url
    if not url.is_file():
        raise FileNotFoundError(f"Missing {field} video: {url}")
    entry["url"] = str(url.resolve())


def build_dataset(
    *,
    jsonl_dir: Path,
    source_image_root: Path,
    source_h5: Path,
    output_dir: Path,
    episodes_per_video: int = 20,
    episode_limit: int | None = None,
) -> dict[str, int]:
    if episodes_per_video <= 0:
        raise ValueError("episodes_per_video must be positive")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary}")

    source_image_root = source_image_root.expanduser().resolve()
    jsonl_files = sorted(jsonl_dir.glob("episode_*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No TacMind-0 episode JSONLs under {jsonl_dir}")
    if episode_limit is not None:
        if episode_limit <= 0:
            raise ValueError("episode_limit must be positive")
        jsonl_files = jsonl_files[:episode_limit]

    episode_rows = []
    for path in jsonl_files:
        episode = int(path.stem.split("_")[-1])
        rows = _read_jsonl(path)
        if len(rows) < 2:
            raise ValueError(f"Episode {episode} has fewer than two rows")
        episode_rows.append((episode, rows))

    temporary.mkdir(parents=True)
    assets_root = temporary / "assets"
    episodes_root = temporary / "episodes"
    episodes_root.mkdir()
    index: dict[str, int] = {}
    video_refs: dict[int, tuple[str, str, int]] = {}
    total_frames = 0

    try:
        with h5py.File(source_h5, "r") as h5:
            required = {"pixels", "ep_len", "ep_offset", "action", "proprio"}
            missing = required.difference(h5.keys())
            if missing:
                raise KeyError(f"HDF5 is missing required datasets: {sorted(missing)}")
            pixels = h5["pixels"]
            if pixels.ndim != 6 or tuple(pixels.shape[1:]) != (
                2,
                8,
                3,
                FRAME_SIZE,
                FRAME_SIZE,
            ):
                raise ValueError(
                    "Expected prepared pixels with shape (N,2,8,3,42,42); "
                    f"got {pixels.shape}"
                )
            if pixels.dtype != np.uint8 or pixels.attrs.get("prepared_layout") != "independent_finger_slot":
                raise ValueError("Expected uint8, independently prepared marker frames")

            h5_lengths = np.asarray(h5["ep_len"][:], dtype=np.int64)
            h5_offsets = np.asarray(h5["ep_offset"][:], dtype=np.int64)
            if len(episode_rows) > len(h5_lengths):
                raise ValueError("More JSONL episodes than aligned HDF5 episodes")

            for group_start in range(0, len(episode_rows), episodes_per_video):
                group = episode_rows[group_start : group_start + episodes_per_video]
                video_index = group_start // episodes_per_video
                video_chunk, video_file = divmod(video_index, VIDEO_FILES_PER_CHUNK)
                video_offset = 0
                group_h5_start = int(h5_offsets[group_start])
                group_h5_end = int(
                    h5_offsets[group_start + len(group) - 1]
                    + h5_lengths[group_start + len(group) - 1]
                )

                for position, (episode, rows) in enumerate(group, start=group_start):
                    if episode != position:
                        raise ValueError(
                            f"Expected contiguous episode {position}, got {episode}"
                        )
                    length = int(h5_lengths[episode])
                    offset = int(h5_offsets[episode])
                    if len(rows) != length:
                        raise ValueError(
                            f"Episode {episode} length mismatch: JSONL={len(rows)}, H5={length}"
                        )
                    actions = np.asarray([row["action"] for row in rows], dtype=np.float32)
                    states = np.asarray([row["state"] for row in rows], dtype=np.float32)
                    if actions.shape != (length, 8) or states.shape != (length, 8):
                        raise ValueError(f"Episode {episode} must contain finite 8D data")
                    if not np.isfinite(actions).all() or not np.isfinite(states).all():
                        raise ValueError(f"Episode {episode} contains non-finite state/action")
                    if not np.array_equal(actions, h5["action"][offset : offset + length]):
                        raise ValueError(f"Episode {episode} actions do not match HDF5 rows")
                    if not np.array_equal(states, h5["proprio"][offset : offset + length]):
                        raise ValueError(f"Episode {episode} states do not match HDF5 rows")

                    marker_left_url = (
                        "videos/observation.images.marker_left/"
                        f"chunk-{video_chunk:03d}/file-{video_file:03d}.mkv"
                    )
                    marker_right_url = (
                        "videos/observation.images.marker_right/"
                        f"chunk-{video_chunk:03d}/file-{video_file:03d}.mkv"
                    )
                    video_refs[episode] = (
                        marker_left_url,
                        marker_right_url,
                        video_offset,
                    )
                    video_offset += length
                    total_frames += length

                left_path = (
                    assets_root
                    / "videos/observation.images.marker_left"
                    / f"chunk-{video_chunk:03d}/file-{video_file:03d}.mkv"
                )
                right_path = (
                    assets_root
                    / "videos/observation.images.marker_right"
                    / f"chunk-{video_chunk:03d}/file-{video_file:03d}.mkv"
                )
                _encode_marker_pair(
                    pixels,
                    group_h5_start,
                    group_h5_end,
                    left_path,
                    right_path,
                )
                print(
                    f"encoded episodes {group[0][0]}-{group[-1][0]} "
                    f"({group_h5_end - group_h5_start} frames)",
                    flush=True,
                )

        for episode, rows in episode_rows:
            marker_left_url, marker_right_url, video_offset = video_refs[episode]
            destination = episodes_root / f"episode_{episode:06d}.jsonl"
            with destination.open("w") as output:
                for t, original in enumerate(rows):
                    row = dict(original)
                    _absolute_camera_url(row, "images_1", source_image_root)
                    _absolute_camera_url(row, "images_2", source_image_root)
                    row["marker_left"] = {
                        "type": "video",
                        "url": marker_left_url,
                        "frame_idx": video_offset + t,
                    }
                    row["marker_right"] = {
                        "type": "video",
                        "url": marker_right_url,
                        "frame_idx": video_offset + t,
                    }
                    output.write(json.dumps(row, separators=(",", ":")) + "\n")
            index[destination.name] = len(rows)

        (episodes_root / "index_cache.json").write_text(
            json.dumps({"data": index}, indent=2) + "\n"
        )
        contract = {
            "source_jsonl_dir": str(jsonl_dir.resolve()),
            "source_camera_root": str(source_image_root),
            "source_marker_h5": str(source_h5.resolve()),
            "source_marker_feature": "pixels[:, finger, current_history_slot]",
            "source_marker_layout": "independent_finger_slot",
            "marker_frame_shape_bgr": [FRAME_SIZE, FRAME_SIZE, 3],
            "marker_frame_codec": "FFV1 lossless Matroska",
            "fps": FPS,
            "history_size": 8,
            "history_stride": 5,
            "episodes": len(episode_rows),
            "frames": total_frames,
            "episode_action_state_alignment": "exact HDF5 equality required per row",
            "camera_urls": "absolute paths into source LeRobot root",
        }
        (temporary / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
        (temporary / "README.md").write_text(
            "# TacMind-0 marker input data\n\n"
            "`marker_left/right` are genuine marker frames extracted from the "
            "aligned LeWM HDF5, not aliases of `tactile_left/right`. Frames are "
            "losslessly encoded at 42x42 BGR, the exact prepared resolution consumed "
            "by the Tac-LeWM tactile encoder. Camera URLs point to the source LeRobot root.\n"
        )
        temporary.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {"episodes": len(episode_rows), "frames": total_frames}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl-dir", type=Path, required=True)
    parser.add_argument("--source-image-root", type=Path, required=True)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes-per-video", type=int, default=20)
    parser.add_argument("--episode-limit", type=int)
    args = parser.parse_args()
    report = build_dataset(
        jsonl_dir=args.jsonl_dir,
        source_image_root=args.source_image_root,
        source_h5=args.source_h5,
        output_dir=args.output_dir,
        episodes_per_video=args.episodes_per_video,
        episode_limit=args.episode_limit,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
