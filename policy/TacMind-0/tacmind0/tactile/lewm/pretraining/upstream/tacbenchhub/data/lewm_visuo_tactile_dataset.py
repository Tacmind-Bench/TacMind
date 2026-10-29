"""Build visuo-tactile LeWM HDF5 from gsmini LeRobot roots.

SWM layout (``ep_len`` / ``ep_offset`` + concatenated columns) so
``swm.data.load_dataset`` can train visuo-JEPA. ``pixels`` is the marker
mosaic; ``tactile`` is an HDF5 hard link to ``pixels``. Tiny-conv pair
loaders still accept legacy ``episode_*`` groups.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import h5py
import numpy as np

from tacbenchhub.data.lewm_marker_dataset import (
    load_actions,
    load_marker_streams,
    load_stream,
    load_vector_column,
    write_preview,
)
from tacbenchhub.data.tactile_history_video import (
    HISTORY_STRIDE,
    REST_FRAMES,
    capped_episode_rows,
    resize_history_mosaic,
)

CONTACT_FORCE_N = 0.25
DEFAULT_PAIR_STRIDE = HISTORY_STRIDE
VISION_KEY = "observation.images.agentview"
WRIST_KEY = "observation.images.wrist"
FORCE_KEY = "observation.force"


def resolve_wrist_coords(
    row: dict,
    *,
    require_wrist: bool,
    dataset: Path,
    episode: int,
) -> object | None:
    """Return LeRobot wrist video coords, or None if optional and absent."""
    if "wrist" in row:
        return row["wrist"]
    if require_wrist:
        raise KeyError(f"{dataset} episode {episode} is missing wrist video")
    return None


def contact_from_force(
    force: np.ndarray,
    threshold: float = CONTACT_FORCE_N,
) -> np.ndarray:
    """Binary contact: either finger L2 force exceeds ``threshold`` (newtons)."""
    array = np.asarray(force, dtype=np.float32)
    if array.ndim == 1:
        if array.shape[0] != 6:
            raise ValueError(f"force must have 6 values, got {array.shape}")
        array = array[None, :]
        squeeze = True
    elif array.ndim == 2 and array.shape[1] == 6:
        squeeze = False
    else:
        raise ValueError(f"force must be (6,) or (T, 6), got {array.shape}")
    left = np.linalg.norm(array[:, :3], axis=-1)
    right = np.linalg.norm(array[:, 3:], axis=-1)
    contact = ((left > float(threshold)) | (right > float(threshold))).astype(np.float32)
    if squeeze:
        return contact[0]
    return contact


def contact_from_episode_force(
    force: np.ndarray,
    threshold: float = CONTACT_FORCE_N,
    rest_frames: int = REST_FRAMES,
) -> np.ndarray:
    """Episode contact from residual wrench vs the start-of-episode rest median.

    Expert demos start unloaded. Raw L2 > 0.25 N would treat a constant gel
    bias as contact on every frame.
    """
    array = np.asarray(force, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 6:
        raise ValueError(f"episode force must be (T, 6), got {array.shape}")
    n_rest = min(max(int(rest_frames), 1), int(array.shape[0]))
    rest = np.median(array[:n_rest], axis=0)
    return contact_from_force(array - rest, threshold=threshold)


def rest_subtracted_wrench(
    force: np.ndarray,
    rest_frames: int = REST_FRAMES,
) -> np.ndarray:
    """Per-episode 6-D residual wrench vs the start-of-episode rest median."""
    array = np.asarray(force, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 6:
        raise ValueError(f"episode force must be (T, 6), got {array.shape}")
    n_rest = min(max(int(rest_frames), 1), int(array.shape[0]))
    rest = np.median(array[:n_rest], axis=0)
    return array - rest


def force_aligned_to_episode(
    force: np.ndarray,
    n_frames: int,
    rest_frames: int = REST_FRAMES,
) -> np.ndarray:
    """Rest-subtract the same truncated window the mix ``contact`` label uses."""
    array = np.asarray(force, dtype=np.float32)
    take = int(n_frames)
    if array.shape[0] < take:
        raise ValueError(f"force length {array.shape[0]} < episode frames {take}")
    return rest_subtracted_wrench(array[:take], rest_frames=rest_frames)


def force_contact_disagreement(
    force: np.ndarray,
    contact: np.ndarray,
    *,
    threshold: float = CONTACT_FORCE_N,
) -> float:
    """Fraction of frames where rest-subtracted wrench L2 disagrees with ``contact``."""
    predicted = contact_from_force(force, threshold=threshold).reshape(-1)
    stored = np.asarray(contact, dtype=np.float32).reshape(-1)
    if predicted.shape != stored.shape:
        raise ValueError(
            f"force contact length {int(predicted.size)} != contact {int(stored.size)}"
        )
    return float(np.mean((predicted > 0.5) != (stored > 0.5)))


def _require_force_matches_contact(
    force: np.ndarray,
    contact: np.ndarray,
    *,
    threshold: float,
    path: Path,
) -> None:
    disagree = force_contact_disagreement(force, contact, threshold=threshold)
    if disagree > 0.0:
        raise ValueError(
            f"{path} rest-subtracted force disagrees with contact on "
            f"{disagree:.2%} of frames; episode order or rest window is wrong"
        )


def load_force(dataset: Path) -> dict[int, np.ndarray]:
    return load_vector_column(dataset, FORCE_KEY, width=6)


def resize_agentview(frame: np.ndarray, out_size: int) -> np.ndarray:
    size = int(out_size)
    if size <= 0:
        raise ValueError(f"out_size must be positive, got {size}")
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HxWx3 agentview, got {image.shape}")
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def side_by_side_preview(
    vision_bgr: np.ndarray,
    left_m: list[np.ndarray],
    right_m: list[np.ndarray],
    frame_index: int,
    out_size: int,
) -> np.ndarray:
    """Agentview | marker mosaic, both square ``out_size`` so heights match."""
    vision = resize_agentview(vision_bgr, out_size)
    tactile = resize_history_mosaic(left_m, right_m, frame_index, out_size=out_size)
    return np.concatenate([vision, tactile], axis=1)


def _aligned_episode_arrays(
    *,
    vision: np.ndarray,
    tactile: np.ndarray,
    action: np.ndarray,
    proprio: np.ndarray,
    contact: np.ndarray,
    wrist: np.ndarray | None = None,
    force: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    arrays = {
        "vision": np.asarray(vision),
        "tactile": np.asarray(tactile),
        "action": np.asarray(action),
        "proprio": np.asarray(proprio),
        "contact": np.asarray(contact),
    }
    if wrist is not None:
        arrays["wrist"] = np.asarray(wrist)
    if force is not None:
        arrays["force"] = np.asarray(force)
    lengths = {key: int(value.shape[0]) for key, value in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"episode length mismatch: {lengths}")
    return arrays


def _write_episode_group(
    handle: h5py.File,
    name: str,
    *,
    vision: np.ndarray,
    tactile: np.ndarray,
    action: np.ndarray,
    proprio: np.ndarray,
    contact: np.ndarray,
    task_id: int,
) -> None:
    """Legacy per-episode groups. New files use SWM ``ep_len``/``ep_offset``."""
    arrays = _aligned_episode_arrays(
        vision=vision,
        tactile=tactile,
        action=action,
        proprio=proprio,
        contact=contact,
    )
    grp = handle.create_group(name)
    vision = arrays["vision"]
    tactile = arrays["tactile"]
    grp.create_dataset(
        "vision", data=vision, compression="gzip", chunks=(1, *vision.shape[1:])
    )
    grp.create_dataset(
        "tactile", data=tactile, compression="gzip", chunks=(1, *tactile.shape[1:])
    )
    grp["pixels"] = grp["tactile"]
    grp.create_dataset("action", data=arrays["action"].astype(np.float32))
    grp.create_dataset("proprio", data=arrays["proprio"].astype(np.float32))
    grp.create_dataset("contact", data=arrays["contact"].astype(np.float32))
    grp.attrs["task_id"] = int(task_id)


def _visuo_episode_payload(
    *,
    vision: np.ndarray,
    tactile: np.ndarray,
    action: np.ndarray,
    proprio: np.ndarray,
    contact: np.ndarray,
    task_id: int,
    wrist: np.ndarray | None = None,
    force: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    arrays = _aligned_episode_arrays(
        vision=vision,
        tactile=tactile,
        action=action,
        proprio=proprio,
        contact=contact,
        wrist=wrist,
        force=force,
    )
    n_frames = int(arrays["vision"].shape[0])
    payload = {
        "pixels": arrays["tactile"],
        "vision": arrays["vision"],
        "action": arrays["action"].astype(np.float32),
        "proprio": arrays["proprio"].astype(np.float32),
        "contact": arrays["contact"].astype(np.float32),
        "task_id": np.full((n_frames,), int(task_id), dtype=np.int32),
    }
    if "wrist" in arrays:
        payload["wrist"] = arrays["wrist"]
    if "force" in arrays:
        payload["force"] = arrays["force"].astype(np.float32)
    return payload


def _is_swm_h5(handle: h5py.File) -> bool:
    return "ep_len" in handle


def _episode_names_and_lengths(handle: h5py.File) -> tuple[list[str], list[int]]:
    if _is_swm_h5(handle):
        lengths = [int(value) for value in handle["ep_len"][:]]
        names = [f"episode_{index:05d}" for index in range(len(lengths))]
        return names, lengths
    names = sorted(name for name in handle.keys() if str(name).startswith("episode_"))
    lengths = [int(handle[name]["contact"].shape[0]) for name in names]
    return names, lengths


def _episode_index(name: str) -> int:
    text = str(name)
    if text.startswith("episode_"):
        return int(text.split("_")[-1])
    return int(text)


def _link_pixels_tactile(root: h5py.Group) -> int:
    """Add missing ``pixels`` ↔ ``tactile`` hard links on one HDF5 group/file."""
    linked = 0
    if "pixels" not in root and "tactile" in root:
        root["pixels"] = root["tactile"]
        linked += 1
    if "tactile" not in root and "pixels" in root:
        root["tactile"] = root["pixels"]
        linked += 1
    return linked


def ensure_pixels_hardlink(h5_path: Path) -> int:
    """Add missing ``pixels`` ↔ ``tactile`` hard links. Returns links added.

    Opens the file read-only first so parallel trainers on an already-linked
    mix HDF5 do not contend for the exclusive HDF5 lock.
    """
    path = Path(h5_path)
    if not path.is_file():
        raise FileNotFoundError(f"visuo-tactile HDF5 not found: {path}")
    with h5py.File(path, "r") as handle:
        if _is_swm_h5(handle):
            needs = "pixels" not in handle or "tactile" not in handle
        else:
            needs = False
            for name in handle:
                if not str(name).startswith("episode_"):
                    continue
                group = handle[name]
                if "pixels" not in group or "tactile" not in group:
                    needs = True
                    break
        if not needs:
            return 0
    linked = 0
    with h5py.File(path, "a") as handle:
        if _is_swm_h5(handle):
            return _link_pixels_tactile(handle)
        for name in handle:
            if not str(name).startswith("episode_"):
                continue
            linked += _link_pixels_tactile(handle[name])
    return linked


def ensure_force_column(
    h5_path: Path,
    sources: list[tuple[Path, int]] | None = None,
    *,
    max_episodes: int = 50,
) -> int:
    """Write rest-subtracted 6-D ``force`` onto an existing SWM mix file.

    Read-first: if ``force`` already matches the frame count, return 0.
    Episode order follows ``sources`` then ``capped_episode_rows``, the same
    as ``build_visuo_tactile_sources``.
    """
    path = Path(h5_path)
    if not path.is_file():
        raise FileNotFoundError(f"visuo-tactile HDF5 not found: {path}")
    with h5py.File(path, "r") as handle:
        if not _is_swm_h5(handle):
            raise ValueError(f"{path} is not swm_hdf5; rebuild instead of patching force")
        frames = int(np.asarray(handle["ep_len"][:]).sum())
        threshold = float(handle.attrs.get("contact_threshold", CONTACT_FORCE_N))
        contact = np.asarray(handle["contact"][:]) if "contact" in handle else None
        if "force" in handle:
            shape = tuple(handle["force"].shape)
            if shape != (frames, 6):
                raise ValueError(f"{path} force shape {shape} != ({frames}, 6)")
            if contact is not None:
                _require_force_matches_contact(
                    np.asarray(handle["force"][:]),
                    contact,
                    threshold=threshold,
                    path=path,
                )
            return 0
        lengths = np.asarray(handle["ep_len"][:], dtype=np.int64)
        rest_frames = int(handle.attrs.get("rest_frames", REST_FRAMES))
    if sources is None:
        from tacbenchhub.assets.paths import project_root
        from tacbenchhub.models.lewm.multitask_sources import dataset_roots

        sources = [(root, task_id) for task_id, root in dataset_roots(project_root())]
    chunks: list[np.ndarray] = []
    index = 0
    for dataset, _task_id in sources:
        forces = load_force(dataset)
        rows = capped_episode_rows(dataset, int(max_episodes))
        for row in rows:
            if index >= int(lengths.size):
                raise ValueError(
                    f"{path} has {int(lengths.size)} episodes but sources produced more"
                )
            episode = int(row["episode_index"])
            chunks.append(
                force_aligned_to_episode(
                    forces[episode],
                    int(lengths[index]),
                    rest_frames=rest_frames,
                )
            )
            index += 1
    if index != int(lengths.size):
        raise ValueError(
            f"force sources produced {index} episodes, hdf5 has {int(lengths.size)}"
        )
    stacked = np.concatenate(chunks, axis=0).astype(np.float32)
    if stacked.shape[0] != frames:
        raise ValueError(f"force frames {stacked.shape[0]} != hdf5 frames {frames}")
    if contact is not None:
        _require_force_matches_contact(
            stacked, contact, threshold=threshold, path=path
        )
    with h5py.File(path, "a") as handle:
        if "force" in handle:
            del handle["force"]
        handle.create_dataset("force", data=stacked)
    return 1


def _set_visuo_attrs(
    h5_path: Path,
    *,
    n_episodes: int,
    out_size: int,
    contact_threshold: float,
    rest_frames: int,
) -> None:
    ensure_pixels_hardlink(h5_path)
    with h5py.File(h5_path, "a") as handle:
        handle.attrs["n_episodes"] = int(n_episodes)
        handle.attrs["out_size"] = int(out_size)
        handle.attrs["mosaic_resize"] = "per_cell"
        handle.attrs["contact_threshold"] = float(contact_threshold)
        handle.attrs["rest_frames"] = int(rest_frames)
        handle.attrs["format"] = "swm_hdf5"


def _resume_episode_count(h5_path: Path, *, resume: bool) -> int:
    """Return the number of complete episodes already present in an HDF5."""
    if not resume or not h5_path.exists():
        return 0
    with h5py.File(h5_path, "r") as handle:
        if "ep_len" not in handle or "ep_offset" not in handle:
            raise ValueError(f"cannot resume {h5_path}: missing ep_len/ep_offset metadata")
        return int(handle["ep_len"].shape[0])


def build_visuo_tactile_sources(
    sources: list[tuple[Path, int]],
    h5_path: Path,
    *,
    out_size: int,
    max_episodes: int | None,
    preview_episodes: int,
    preview_dir: Path,
    contact_threshold: float = CONTACT_FORCE_N,
    rest_frames: int = REST_FRAMES,
    require_wrist: bool = True,
    resume: bool = False,
) -> dict[str, int]:
    from stable_worldmodel.data.formats.hdf5 import HDF5Writer

    if not sources:
        raise ValueError("at least one LeRobot dataset is required")
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    existing_episodes = _resume_episode_count(h5_path, resume=bool(resume))
    if existing_episodes:
        print(f"resuming {h5_path} after {existing_episodes} complete episodes", flush=True)
    written_episodes = 0
    written_frames = 0
    skip_episodes = existing_episodes
    mode = "append" if existing_episodes else "overwrite"
    with HDF5Writer(h5_path, mode=mode) as writer:
        for dataset, task_id in sources:
            cache: dict[Path, list[np.ndarray]] = {}
            actions = load_actions(dataset)
            forces = load_force(dataset)
            rows = capped_episode_rows(dataset, max_episodes)
            remaining_previews = int(preview_episodes)
            for row in rows:
                if skip_episodes:
                    skip_episodes -= 1
                    continue
                episode = int(row["episode_index"])
                length = int(row["length"])
                left_m, right_m = load_marker_streams(cache, dataset, row, length)
                if "agentview" not in row:
                    raise KeyError(f"{dataset} episode {episode} is missing agentview video")
                vision_frames = load_stream(
                    cache, dataset, VISION_KEY, row["agentview"], length
                )
                wrist_coords = resolve_wrist_coords(
                    row,
                    require_wrist=bool(require_wrist),
                    dataset=dataset,
                    episode=episode,
                )
                wrist_frames = None
                if wrist_coords is not None:
                    wrist_frames = load_stream(
                        cache, dataset, WRIST_KEY, wrist_coords, length
                    )
                action, proprio = actions[episode]
                force = forces[episode]
                lengths = [
                    len(left_m),
                    len(right_m),
                    len(vision_frames),
                    len(action),
                    len(proprio),
                    len(force),
                ]
                if wrist_frames is not None:
                    lengths.append(len(wrist_frames))
                n_frames = min(lengths)
                left_m = left_m[:n_frames]
                right_m = right_m[:n_frames]
                vision_frames = vision_frames[:n_frames]
                contact = contact_from_episode_force(
                    force[:n_frames],
                    threshold=contact_threshold,
                    rest_frames=rest_frames,
                )
                residual = rest_subtracted_wrench(
                    force[:n_frames], rest_frames=rest_frames
                )
                tactile = np.stack(
                    [
                        resize_history_mosaic(left_m, right_m, index, out_size=out_size)
                        for index in range(n_frames)
                    ]
                )
                vision = np.stack(
                    [resize_agentview(frame, out_size) for frame in vision_frames]
                )
                wrist = None
                if wrist_frames is not None:
                    wrist = np.stack(
                        [
                            resize_agentview(frame, out_size)
                            for frame in wrist_frames[:n_frames]
                        ]
                    )
                print(
                    f"task {int(task_id):02d} episode {episode:02d} "
                    f"frames={n_frames} visuo-tactile",
                    flush=True,
                )
                writer.write_episode(
                    _visuo_episode_payload(
                        vision=vision,
                        tactile=tactile,
                        action=action[:n_frames],
                        proprio=proprio[:n_frames],
                        contact=contact,
                        task_id=int(task_id),
                        wrist=wrist,
                        force=residual,
                    )
                )
                written_episodes += 1
                written_frames += n_frames
                if remaining_previews > 0:
                    preview = [
                        np.concatenate([vision[index], tactile[index]], axis=1)
                        for index in range(n_frames)
                    ]
                    write_preview(
                        preview_dir / f"task_{int(task_id):02d}_episode_{episode:03d}.mp4",
                        preview,
                    )
                    remaining_previews -= 1
    if skip_episodes:
        raise ValueError(
            f"cannot resume {h5_path}: existing file has {existing_episodes} episodes, "
            "but the source set contains fewer episodes"
        )
    total_episodes = existing_episodes + written_episodes
    _set_visuo_attrs(
        h5_path,
        n_episodes=total_episodes,
        out_size=out_size,
        contact_threshold=contact_threshold,
        rest_frames=rest_frames,
    )
    print(
        f"wrote {h5_path} episodes={written_episodes} frames={written_frames}",
        flush=True,
    )
    return {"episodes": written_episodes, "frames": written_frames}


def list_pair_index(
    h5_path: Path,
    *,
    stride: int = 1,
) -> list[tuple[str, int]]:
    """Return ``(episode_name, t)`` for every ``(t, t+stride)`` pair."""
    step = int(stride)
    if step <= 0:
        raise ValueError(f"stride must be positive, got {step}")
    pairs: list[tuple[str, int]] = []
    with h5py.File(h5_path, "r") as handle:
        names, lengths = _episode_names_and_lengths(handle)
        if not names:
            raise ValueError(f"no episodes in {h5_path}")
        for name, length in zip(names, lengths, strict=True):
            if length <= step:
                continue
            pairs.extend((name, t) for t in range(length - step))
    if not pairs:
        raise ValueError(f"no stride-{step} pairs in {h5_path}")
    return pairs


def split_pair_index(
    pairs: list[tuple[str, int]],
    *,
    val_fraction: float = 0.1,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Hold out later episodes (or the tail of a single episode)."""
    if not pairs:
        raise ValueError("no pairs to split")
    frac = float(val_fraction)
    if not 0.0 <= frac < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {frac}")
    if frac == 0.0:
        return list(pairs), []
    names = list(dict.fromkeys(name for name, _index in pairs))
    if len(names) == 1:
        if len(pairs) < 2:
            return list(pairs), []
        n_val = min(max(1, int(round(len(pairs) * frac))), len(pairs) - 1)
        return list(pairs[:-n_val]), list(pairs[-n_val:])
    n_val = min(max(1, int(round(len(names) * frac))), len(names) - 1)
    val_names = set(names[-n_val:])
    train = [pair for pair in pairs if pair[0] not in val_names]
    val = [pair for pair in pairs if pair[0] in val_names]
    return train, val


def _swm_row(handle: h5py.File, name: str, index: int) -> int:
    return int(handle["ep_offset"][_episode_index(name)]) + int(index)


def _column(handle: h5py.File, name: str, key: str):
    root = handle if _is_swm_h5(handle) else handle[name]
    if key == "tactile" and "tactile" not in root:
        return root["pixels"]
    return root[key]


def contact_bce_pos_weight(
    h5_path: Path,
    pairs: list[tuple[str, int]],
    *,
    stride: int = 1,
) -> float:
    """``(1-p)/p`` for BCE ``pos_weight`` from ``next_contact`` of ``pairs``."""
    if not pairs:
        return 1.0
    step = int(stride)
    grouped: dict[str, list[int]] = {}
    for name, index in pairs:
        grouped.setdefault(name, []).append(int(index))
    positive = 0.0
    total = 0.0
    with h5py.File(h5_path, "r") as handle:
        swm = _is_swm_h5(handle)
        for name, indices in grouped.items():
            contact = np.asarray(_column(handle, name, "contact"))
            offset = int(handle["ep_offset"][_episode_index(name)]) if swm else 0
            idx = np.asarray(indices, dtype=np.int64) + step + offset
            positive += float(contact[idx].sum())
            total += float(idx.size)
    rate = positive / total
    rate = min(max(rate, 1e-3), 1.0 - 1e-3)
    return float((1.0 - rate) / rate)


def load_pair(
    h5_path: Path,
    name: str,
    index: int,
    handle: h5py.File | None = None,
    *,
    stride: int = 1,
) -> dict[str, np.ndarray]:
    step = int(stride)
    if step <= 0:
        raise ValueError(f"stride must be positive, got {step}")
    own = handle is None
    file_handle = h5py.File(h5_path, "r") if own else handle
    assert file_handle is not None
    try:
        if _is_swm_h5(file_handle):
            row = _swm_row(file_handle, name, index)
            next_row = _swm_row(file_handle, name, int(index) + step)
        else:
            row = int(index)
            next_row = int(index) + step
        vision = _column(file_handle, name, "vision")
        tactile = _column(file_handle, name, "tactile")
        return {
            "vision": np.asarray(vision[row]),
            "tactile": np.asarray(tactile[row]),
            "next_vision": np.asarray(vision[next_row]),
            "next_tactile": np.asarray(tactile[next_row]),
            "action": np.asarray(_column(file_handle, name, "action")[row]),
            "proprio": np.asarray(_column(file_handle, name, "proprio")[row]),
            "next_contact": np.asarray(_column(file_handle, name, "contact")[next_row]),
        }
    finally:
        if own:
            file_handle.close()
