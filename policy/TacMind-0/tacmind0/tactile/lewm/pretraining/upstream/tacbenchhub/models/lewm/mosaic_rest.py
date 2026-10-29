"""Per-episode GelSight rest subtraction for 4x4 mosaics (WM-10 / WM-11–13).

Do not drop RGB. ``I_0`` is the median of the first ``REST_FRAMES`` current
L7|R7 cells, tiled onto every slot of that finger.

- ``subtract``: ``I_t - I_0``
- ``divide``: ``(I_t - I_0) / (|I_0| + eps)`` (per-pixel; amplifies dark gel)
- ``log``: ``log(I_t + eps) - log(I_0 + eps)`` (multiplicative lighting)
- ``smooth_div``: ``(I_t - I_0) / (blur(I_0) + eps)`` (low-frequency lamp only)
- ``highpass``: ``(I_t - I_0) - blur(I_t - I_0)`` (keep marker-scale change)

Signed residuals map to ``[0, 1]`` with 0.5 = no deformation. Camera residual
is unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tacbenchhub.data.tactile_history_video import (
    GRID_COLS,
    GRID_ROWS,
    HISTORY_FRAMES,
    REST_FRAMES,
    SIDE_COLS,
    _tile_sides,
)

MOSAIC_REST_SUBS = ("off", "subtract", "divide", "log", "smooth_div", "highpass")
MOSAIC_REST_ALIASES = {
    "false": "off",
    "off": "off",
    "none": "off",
    "0": "off",
    "true": "subtract",
    "on": "subtract",
    "sub": "subtract",
    "subtract": "subtract",
    "diff": "subtract",
    "divide": "divide",
    "div": "divide",
    "normalized": "divide",
    "log": "log",
    "log_ratio": "log",
    "ratio": "log",
    "smooth_div": "smooth_div",
    "blur_div": "smooth_div",
    "blur_divide": "smooth_div",
    "highpass": "highpass",
    "hp": "highpass",
}
REST_SUB_EPS = 1.0 / 255.0


def parse_mosaic_rest_sub(value: bool | str | None) -> str:
    if value is None or value is False:
        return "off"
    if value is True:
        return "subtract"
    key = str(value).strip().lower()
    if key not in MOSAIC_REST_ALIASES:
        raise ValueError(
            "mosaic_rest_sub must be off, subtract, divide, log, "
            f"smooth_div, or highpass, got {value!r}"
        )
    return MOSAIC_REST_ALIASES[key]


def mosaic_l7_r7_cells(mosaic_hwc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Current left|right cells (L7, R7) from a 4x4 mosaic."""
    image = np.asarray(mosaic_hwc)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected HWC mosaic, got {tuple(image.shape)}")
    height, width = int(image.shape[0]), int(image.shape[1])
    if height % GRID_ROWS != 0 or width % GRID_COLS != 0:
        raise ValueError(
            f"mosaic must divide {GRID_ROWS}x{GRID_COLS}, got {height}x{width}"
        )
    cell_h, cell_w = height // GRID_ROWS, width // GRID_COLS
    left = image[3 * cell_h : 4 * cell_h, 1 * cell_w : 2 * cell_w]
    right = image[3 * cell_h : 4 * cell_h, 3 * cell_w : 4 * cell_w]
    return np.ascontiguousarray(left), np.ascontiguousarray(right)


def tile_finger_rest(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Repeat rest L|R into every mosaic slot of that finger."""
    left_u8 = np.asarray(left, dtype=np.uint8)
    right_u8 = np.asarray(right, dtype=np.uint8)
    return _tile_sides(
        [left_u8] * HISTORY_FRAMES,
        [right_u8] * HISTORY_FRAMES,
    )


def episode_rest_mosaics_hwc(
    h5_path: Path,
    *,
    rest_frames: int | None = None,
) -> np.ndarray:
    """One rest mosaic per episode from the median of early current L7|R7 cells.

    At episode start the current cells are the unloaded gel (same window the
    contact label uses). Read-only: does not write the mix HDF5.
    """
    import h5py

    path = Path(h5_path)
    with h5py.File(path, "r") as handle:
        if "pixels" not in handle or "ep_offset" not in handle or "ep_len" not in handle:
            raise KeyError(f"{path} needs pixels, ep_offset, and ep_len")
        offsets = np.asarray(handle["ep_offset"][:], dtype=np.int64)
        lengths = np.asarray(handle["ep_len"][:], dtype=np.int64)
        n_rest = int(rest_frames) if rest_frames is not None else int(
            handle.attrs.get("rest_frames", REST_FRAMES)
        )
        n_rest = max(int(n_rest), 1)
        pixels = handle["pixels"]
        rest = []
        for offset, length in zip(offsets, lengths, strict=True):
            take = min(n_rest, int(length))
            if take <= 0:
                raise ValueError("episode length must be positive")
            chunk = np.asarray(pixels[int(offset) : int(offset) + take])
            left_cells = []
            right_cells = []
            for frame in chunk:
                left, right = mosaic_l7_r7_cells(frame)
                left_cells.append(left)
                right_cells.append(right)
            left_rest = np.median(np.stack(left_cells, axis=0), axis=0).astype(np.uint8)
            right_rest = np.median(np.stack(right_cells, axis=0), axis=0).astype(np.uint8)
            rest.append(tile_finger_rest(left_rest, right_rest))
    return np.stack(rest, axis=0)


def _spatial_blur(image: Any) -> Any:
    """Odd average-pool kernel ~ one ViT patch so lamps vary slower than markers."""
    import torch

    tensor = torch.as_tensor(image)
    squeezed = False
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
        squeezed = True
    if tensor.ndim != 4:
        raise ValueError(f"expected CHW or NCHW, got {tuple(tensor.shape)}")
    _, _, height, _width = tensor.shape
    kernel = max(3, (int(height) // 14) | 1)
    if kernel % 2 == 0:
        kernel += 1
    pad = kernel // 2
    blurred = torch.nn.functional.avg_pool2d(
        tensor,
        kernel_size=kernel,
        stride=1,
        padding=pad,
        count_include_pad=False,
    )
    if blurred.shape[-2:] != tensor.shape[-2:]:
        blurred = torch.nn.functional.interpolate(
            blurred,
            size=tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    return blurred.squeeze(0) if squeezed else blurred


def apply_mosaic_rest_sub(
    pixels: Any,
    rest_hwc: Any,
    *,
    mode: str,
    eps: float = REST_SUB_EPS,
) -> Any:
    """Apply rest-sub to NCHW unit-interval mosaics. 0.5 means no deformation."""
    import torch

    from tacbenchhub.models.lewm.visuo_jepa import resize_mosaic_nchw

    chosen = parse_mosaic_rest_sub(mode)
    if chosen == "off":
        return pixels
    tensor = torch.as_tensor(pixels)
    squeezed = False
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
        squeezed = True
    if tensor.ndim != 4 or tensor.shape[1] != 3:
        raise ValueError(f"expected (T,3,H,W) mosaic, got {tuple(tensor.shape)}")
    rest = torch.as_tensor(rest_hwc)
    as_uint8 = rest.dtype == torch.uint8
    if rest.ndim == 3 and rest.shape[-1] == 3:
        rest = rest.permute(2, 0, 1).contiguous()
    if rest.ndim != 3 or rest.shape[0] != 3:
        raise ValueError(f"expected rest HWC or CHW, got {tuple(rest.shape)}")
    rest = rest.float()
    if as_uint8 or float(rest.max()) > 1.5:
        rest = rest.div(255.0)
    if rest.shape[-2:] != tensor.shape[-2:]:
        rest = resize_mosaic_nchw(rest.unsqueeze(0), int(tensor.shape[-1])).squeeze(0)
    rest = rest.to(device=tensor.device, dtype=tensor.dtype)
    frames = tensor.float()
    eps_f = float(eps)
    if chosen == "subtract":
        delta = frames - rest
    elif chosen == "divide":
        delta = (frames - rest) / (rest.abs() + eps_f)
    elif chosen == "log":
        delta = (frames.clamp(min=0.0) + eps_f).log() - (rest.clamp(min=0.0) + eps_f).log()
    elif chosen == "smooth_div":
        lamp = _spatial_blur(rest)
        delta = (frames - rest) / (lamp.abs() + eps_f)
    elif chosen == "highpass":
        residual = frames - rest
        delta = residual - _spatial_blur(residual)
    else:
        raise ValueError(f"unsupported mosaic_rest_sub {chosen!r}")
    delta = delta.clamp(-1.0, 1.0)
    out = ((delta + 1.0) * 0.5).clamp(0.0, 1.0)
    return out.squeeze(0) if squeezed else out


class EpisodeRestSubtract:
    """SWM dataset wrapper: subtract per-episode gel rest from ``pixels``."""

    def __init__(self, dataset: Any, rest_hwc: np.ndarray, mode: str) -> None:
        chosen = parse_mosaic_rest_sub(mode)
        if chosen == "off":
            raise ValueError("EpisodeRestSubtract requires a rest-sub mode, not off")
        rest = np.asarray(rest_hwc)
        if rest.ndim != 4 or rest.shape[-1] != 3:
            raise ValueError(f"expected rest (E,H,W,3), got {tuple(rest.shape)}")
        self.dataset = dataset
        self.rest_hwc = np.ascontiguousarray(rest)
        self.mode = chosen

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        clips = getattr(self.dataset, "clip_indices", None)
        if clips is None:
            raise TypeError("dataset has no clip_indices; cannot rest-subtract")
        episode, _start = clips[index]
        if "pixels" not in sample:
            raise KeyError("pixels not in sample")
        sample["pixels"] = apply_mosaic_rest_sub(
            sample["pixels"],
            self.rest_hwc[int(episode)],
            mode=self.mode,
        )
        return sample

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)


def maybe_wrap_rest_sub(
    dataset: Any,
    h5_path: Path,
    mode: str | None,
) -> Any:
    chosen = parse_mosaic_rest_sub(mode)
    if chosen == "off":
        return dataset
    rest = episode_rest_mosaics_hwc(h5_path)
    return EpisodeRestSubtract(dataset, rest, chosen)
