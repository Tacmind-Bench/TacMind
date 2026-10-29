"""I/O helpers for LeWM runs that use only the current camera frame.

The world-model sample still contains the complete tactile/action window, but
``vision_history=current`` only consumes one camera frame.  This wrapper reads
only the one needed row from a local SWM HDF5 dataset and returns a compact
one-frame camera tensor; the model restores the original temporal slot after
DINO encoding.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch


class CurrentFrameVisionDataset:
    """Proxy an SWM HDF5 reader and skip unused historical camera reads.

    ``base`` is intentionally duck-typed so the first-party package does not
    import the optional ``stable_worldmodel`` dependency on CPU-only paths.
    The proxy forwards dataset metadata and transforms, and implements the
    same clip sampling contract as ``stable_worldmodel.data.Dataset``.
    """

    def __init__(
        self,
        base: Any,
        *,
        vision_keys: Iterable[str],
        current_index: int,
        compact_current: bool = True,
    ) -> None:
        self.base = base
        self.vision_keys = frozenset(str(key) for key in vision_keys)
        self.current_index = int(current_index)
        self.compact_current = bool(compact_current)
        if self.current_index < 0:
            raise ValueError(f"current_index must be non-negative, got {current_index}")
        self.clip_indices = base.clip_indices

    @property
    def transform(self) -> Any:
        return self.base.transform

    @transform.setter
    def transform(self, value: Any) -> None:
        self.base.transform = value

    def __getattr__(self, name: str) -> Any:
        # Keep get_dim/get_col_data and any backend metadata available to the
        # existing trainer without copying or reimplementing the reader.
        return getattr(self.base, name)

    def __len__(self) -> int:
        return len(self.clip_indices)

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict[str, Any]:
        base = self.base
        base._open()
        global_start = int(base.offsets[ep_idx]) + int(start)
        global_end = int(base.offsets[ep_idx]) + int(end)
        raw_length = int(end) - int(start)
        if raw_length <= 0:
            raise ValueError(f"empty sample slice: start={start}, end={end}")
        frame_offset = self.current_index * int(base.frameskip)
        if frame_offset >= raw_length:
            raise ValueError(
                "current vision index is outside the sample: "
                f"index={self.current_index}, frameskip={base.frameskip}, "
                f"raw_length={raw_length}"
            )

        steps: dict[str, Any] = {}
        for col in base._keys:
            source = base._cache if col in base._cache else base.h5_file
            if col in self.vision_keys:
                # Read one HDF5 chunk. Compact mode avoids allocating,
                # transforming, and transferring unused historical frames.
                sample = np.asarray(source[col][global_start + frame_offset])
                if self.compact_current:
                    data = sample[None, ...]
                else:
                    data = np.zeros((raw_length, *sample.shape), dtype=sample.dtype)
                    data[frame_offset] = sample
                    data = data[:: base.frameskip]
            else:
                data = source[col][global_start:global_end]
                if col != "action":
                    data = data[:: base.frameskip]

            if data.dtype == np.object_ or data.dtype.kind in ("S", "U"):
                value = data[0] if len(data) > 0 else b""
                steps[col] = value.decode() if isinstance(value, bytes) else value
            else:
                tensor = torch.from_numpy(np.asarray(data))
                if tensor.ndim == 4 and tensor.shape[-1] in (1, 3):
                    tensor = tensor.permute(0, 3, 1, 2)
                steps[col] = tensor

        return base.transform(steps) if base.transform else steps

    def __getitem__(self, idx: int) -> dict[str, Any]:
        total = len(self.clip_indices)
        max_attempts = min(16, total) if total > 0 else 1
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try_idx = (idx + attempt) % total
            ep_idx, start = self.clip_indices[try_idx]
            try:
                steps = self._load_slice(ep_idx, start, start + self.base.span)
            except Exception as exc:  # pragma: no cover - defensive backend retry
                last_exc = exc
                continue
            if "action" in steps:
                steps["action"] = steps["action"].reshape(self.base.num_steps, -1)
            return steps
        raise RuntimeError(
            f"Failed to load sample idx={idx} after {max_attempts} attempts; "
            f"last error: {last_exc}"
        )


def maybe_wrap_current_vision_dataset(
    dataset: Any,
    *,
    enabled: bool,
    vision_keys: Iterable[str],
    history_size: int,
    num_steps: int,
    compact_current: bool = True,
) -> Any:
    """Wrap local SWM HDF5 readers when current-frame I/O is requested."""
    if not enabled:
        return dataset
    if not hasattr(dataset, "_keys") or not hasattr(dataset, "_open"):
        return dataset
    steps = int(num_steps)
    if steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    current_index = min(max(int(history_size), 1), steps) - 1
    available = set(dataset._keys)
    selected = tuple(key for key in vision_keys if key in available)
    if not selected:
        return dataset
    return CurrentFrameVisionDataset(
        dataset,
        vision_keys=selected,
        current_index=current_index,
        compact_current=compact_current,
    )
