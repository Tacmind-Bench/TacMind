"""Tac-LeWM prepared-uint8 contract: BGR -> AREA 56 -> bilinear 42 -> round -> /255."""

import json
from collections import OrderedDict, deque
from collections.abc import Sequence
from numbers import Integral
from typing import Any

import cv2
import numpy as np
import torch
from torch.nn import functional as F

HISTORY = 8
STRIDE = 5
HISTORY_SPAN = (HISTORY - 1) * STRIDE


def history_indices(frame_index: int) -> list[int]:
    if (
        isinstance(frame_index, bool)
        or not isinstance(frame_index, Integral)
        or frame_index < 0
    ):
        raise ValueError("frame_index must be a non-negative integer")
    return [max(0, frame_index - HISTORY_SPAN + STRIDE * i) for i in range(HISTORY)]


def validate_marker_frame(frame: Any) -> np.ndarray:
    value = np.asarray(frame)
    if (
        value.dtype != np.uint8
        or value.ndim != 3
        or value.shape[-1] != 3
        or min(value.shape[:2]) == 0
    ):
        raise ValueError("Marker frames must be nonempty HWC uint8 with three channels")
    return value


def _prepare_marker_frame(frame: Any, *, color: str) -> np.ndarray:
    value = validate_marker_frame(frame)
    if color == "rgb":
        value = value[..., ::-1]
    # HDF5-derived training videos may already contain the exact Tac-LeWM 42x42
    # prepared marker frame. Keep it unchanged instead of blurring it through
    # the native-frame AREA56 -> bilinear42 path a second time.
    if value.shape[:2] == (42, 42):
        return np.ascontiguousarray(value)
    return cv2.resize(value, (56, 56), interpolation=cv2.INTER_AREA)


def _normalize_marker_cells(cells: Sequence[np.ndarray]) -> torch.Tensor:
    """Normalize an equal-size batch, preserving already prepared 42px pixels."""
    pixels = torch.from_numpy(np.stack(cells)).permute(0, 3, 1, 2).float()
    if pixels.shape[-2:] != (42, 42):
        pixels = F.interpolate(
            pixels, size=(42, 42), mode="bilinear", align_corners=False
        )
    return pixels.clamp(0, 255).round() / 255.0


def _prepare_frames(frames: Sequence[Any], *, color: str) -> torch.Tensor:
    """Apply the shared Tac-LeWM resize, quantization and color contract to a batch."""
    if color not in ("rgb", "bgr"):
        raise ValueError("color must be rgb or bgr")
    cells = [_prepare_marker_frame(frame, color=color) for frame in frames]
    if all(cell.shape == cells[0].shape for cell in cells):
        return _normalize_marker_cells(cells)
    # Prepared 42px cells and native frames (resized to 56px) may coexist,
    # including across fingers. Batch only equal sizes, then restore frame order.
    prepared = torch.empty((len(cells), 3, 42, 42), dtype=torch.float32)
    for size in (42, 56):
        indices = [index for index, cell in enumerate(cells) if cell.shape[0] == size]
        if not indices:
            continue
        prepared[indices] = _normalize_marker_cells([cells[index] for index in indices])
    return prepared


def prepare_frame_pair(left: Any, right: Any, *, color: str = "rgb") -> torch.Tensor:
    """Prepare one current left/right pair with the same Tac-LeWM pixel contract."""
    return _prepare_frames((left, right), color=color)


def prepare_history(
    left: Sequence[Any], right: Sequence[Any], *, color: str = "rgb"
) -> torch.Tensor:
    if len(left) != HISTORY or len(right) != HISTORY:
        raise ValueError(
            "Exactly eight oldest-to-current marker frames per finger are required"
        )
    frames = [frame for history in (left, right) for frame in history]
    return _prepare_frames(frames, color=color).reshape(2, HISTORY, 3, 42, 42)


class PreparedMarkerCache:
    """Bounded, worker-local cache of deterministic RGB marker preprocessing.

    Keys identify immutable frames within one dataset or episode. Returned
    tensors are read-only; callers stack them into independent sample tensors.
    """

    def __init__(self, capacity: int = 256) -> None:
        if type(capacity) is not int or capacity < 0:
            raise ValueError("marker cache capacity must be a nonnegative integer")
        self.capacity = capacity
        self.entries = OrderedDict()

    def fork(self):
        result = PreparedMarkerCache(self.capacity)
        result.entries = self.entries.copy()
        return result

    def get_many(self, keys, decode):
        unique = list(dict.fromkeys(keys))
        found = {}
        for key in unique:
            if key in self.entries:
                found[key] = self.entries[key]
                self.entries.move_to_end(key)
        missing = [key for key in unique if key not in found]
        if missing:
            frames = [frame for key in missing for frame in decode(key)]
            prepared = _prepare_frames(frames, color="rgb").reshape(-1, 2, 3, 42, 42)
            for key, pair in zip(missing, prepared, strict=True):
                # Own each allocation: a slice must not retain the entire miss batch.
                found[key] = pair.clone() if self.capacity else pair
                if self.capacity:
                    self.entries[key] = found[key]
                    while len(self.entries) > self.capacity:
                        self.entries.popitem(last=False)
        return [found[key] for key in keys]


class LoadTactileHistory:
    def __init__(self, image_dir: str, cache_size: int = 256) -> None:
        self.image_dir = image_dir
        self.cache = PreparedMarkerCache(cache_size)

    def __call__(self, data: dict) -> dict:
        from tacmind0.data.transforms import LoadImages

        t = data["meta_data"]["frame_index"]
        indices = history_indices(t)
        loader = LoadImages(
            ["marker_left", "marker_right"], self.image_dir, require_rgb=True
        )
        pairs = self.cache.get_many(
            [data["raw_lines"][index] for index in indices],
            lambda line: loader(json.loads(line))["images"],
        )
        data["tactile_pixel_values"] = torch.stack(pairs, dim=1).unsqueeze(0)
        return data


class MarkerHistory:
    """One instance per simulation lane; append every control tick, not policy request."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.frames = deque(maxlen=HISTORY_SPAN + 1)
        self.prepared = PreparedMarkerCache(HISTORY_SPAN + 1)
        self.frame_index = -1

    def append(self, left: np.ndarray, right: np.ndarray, *, frame_index: int) -> None:
        history_indices(frame_index)
        if frame_index != self.frame_index + 1:
            raise ValueError(
                "Marker history requires every consecutive control tick; reset at episode start"
            )
        left, right = validate_marker_frame(left), validate_marker_frame(right)
        self.frames.append((left.copy(), right.copy()))
        self.frame_index = frame_index
        # The earliest sampled frame was just touched, so ordinary LRU would
        # retain it after expiry and evict a newer frame needed by another phase.
        self.prepared.entries.pop(frame_index - HISTORY_SPAN - 1, None)

    def selected(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if not self.frames:
            raise ValueError("Empty marker history")
        start = max(0, self.frame_index - HISTORY_SPAN)
        indices = [i - start for i in history_indices(self.frame_index)]
        return tuple([[self.frames[i][side] for i in indices] for side in (0, 1)])

    def tensor(self) -> torch.Tensor:
        if not self.frames:
            raise ValueError("Empty marker history")
        start = max(0, self.frame_index - HISTORY_SPAN)
        pairs = self.prepared.get_many(
            history_indices(self.frame_index), lambda index: self.frames[index - start]
        )
        return torch.stack(pairs, dim=1)

    def current_tensor(self) -> torch.Tensor:
        """Share the latest prepared pair with the Tac-FRS stream encoder."""
        if not self.frames:
            raise ValueError("Empty marker history")
        return self.prepared.get_many([self.frame_index], lambda _: self.frames[-1])[0]
