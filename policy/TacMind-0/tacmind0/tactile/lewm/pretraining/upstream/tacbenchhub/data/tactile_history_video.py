"""Build GelSight tactile history mosaics and dense optical-flow videos."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

HISTORY_FRAMES = 8
HISTORY_STRIDE = 5
GRID_COLS = 4
GRID_ROWS = 4
SIDE_COLS = 2
DEFAULT_FPS = 20.0
FLOW_MAG_CLIP = 5.0
CAMERA_FLOW_MAG_CLIP = 12.0
FLOW_OUTLIER_PX = 12.0
MARKER_SPLAT_SIGMA = 14.0
REST_FRAMES = 5


def history_indices(
    frame_index: int,
    *,
    n: int = HISTORY_FRAMES,
    stride: int = HISTORY_STRIDE,
) -> list[int]:
    """Return ``n`` indices ending at ``frame_index``, spaced by ``stride``.

    Missing history (episode start) repeats frame 0. Newest cell is current.
    """
    if n <= 0 or stride <= 0:
        raise ValueError(f"n and stride must be positive, got n={n} stride={stride}")
    t = max(0, int(frame_index))
    return [max(0, t - (n - 1 - slot) * int(stride)) for slot in range(int(n))]


def _bgr_uint8(frame: np.ndarray) -> np.ndarray:
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 BGR frame, got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def mosaic_cell_size(out_size: int) -> int:
    """Square edge of one 4x4 mosaic cell. ``out_size`` must divide evenly."""
    size = int(out_size)
    if size <= 0:
        raise ValueError(f"out_size must be positive, got {size}")
    if size % GRID_COLS != 0:
        raise ValueError(
            f"out_size must be divisible by {GRID_COLS} so each cell is square, got {size}"
        )
    return size // GRID_COLS


def resize_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """``INTER_AREA`` resize of one gel frame. No neighboring cells."""
    image = _bgr_uint8(frame)
    out_w, out_h = int(width), int(height)
    if out_w <= 0 or out_h <= 0:
        raise ValueError(f"resize size must be positive, got {out_w}x{out_h}")
    if image.shape[0] == out_h and image.shape[1] == out_w:
        return image
    return cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)


def stacked_mosaic_cell_hw(out_size: int) -> tuple[int, int]:
    """Cell ``(h, w)`` for RGB-on-top / marker-below inside a square ``out_size``."""
    size = int(out_size)
    stacked_rows = GRID_ROWS * 2
    if size <= 0:
        raise ValueError(f"out_size must be positive, got {size}")
    if size % stacked_rows != 0 or size % GRID_COLS != 0:
        raise ValueError(
            f"stacked rgb|marker out_size must be divisible by {stacked_rows} "
            f"and {GRID_COLS}, got {size}"
        )
    return size // stacked_rows, size // GRID_COLS


def _place_side_cells(
    mosaic: np.ndarray,
    cells: list[np.ndarray],
    *,
    col_offset: int,
) -> None:
    height, width = cells[0].shape[:2]
    for slot, cell in enumerate(cells):
        row, local_col = divmod(slot, SIDE_COLS)
        col = col_offset + local_col
        mosaic[row * height : (row + 1) * height, col * width : (col + 1) * width] = cell


def _tile_sides(
    left_cells: list[np.ndarray],
    right_cells: list[np.ndarray],
    dest: np.ndarray | None = None,
) -> np.ndarray:
    """Left finger | right finger into a 4x4 raster (or a matching ``dest`` view)."""
    height, width = left_cells[0].shape[:2]
    if dest is None:
        dest = np.empty((height * GRID_ROWS, width * GRID_COLS, 3), dtype=np.uint8)
    _place_side_cells(dest, left_cells, col_offset=0)
    _place_side_cells(dest, right_cells, col_offset=SIDE_COLS)
    return dest


def _history_cells(
    left_frames: list[np.ndarray],
    right_frames: list[np.ndarray],
    frame_index: int,
    *,
    n: int,
    stride: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if len(left_frames) != len(right_frames) or not left_frames:
        raise ValueError("left_frames and right_frames must be non-empty and equal length")
    last = len(left_frames) - 1
    if frame_index < 0 or frame_index > last:
        raise ValueError(f"frame_index {frame_index} out of range 0..{last}")
    if n != GRID_ROWS * SIDE_COLS:
        raise ValueError(f"n must be {GRID_ROWS * SIDE_COLS} for a 4x4 left/right grid")
    indices = history_indices(frame_index, n=n, stride=stride)
    left_cells = [_bgr_uint8(left_frames[index]) for index in indices]
    right_cells = [_bgr_uint8(right_frames[index]) for index in indices]
    return left_cells, right_cells


def tile_tactile_history(
    left_frames: list[np.ndarray],
    right_frames: list[np.ndarray],
    frame_index: int,
    *,
    n: int = HISTORY_FRAMES,
    stride: int = HISTORY_STRIDE,
) -> np.ndarray:
    """Seamless 4x4 mosaic: left finger | right finger, oldest to current.

    Layout (2 columns per finger, 4 rows)::

        L0 L1 | R0 R1
        L2 L3 | R2 R3
        L4 L5 | R4 R5
        L6 L7 | R6 R7
    """
    left_cells, right_cells = _history_cells(
        left_frames, right_frames, frame_index, n=n, stride=stride
    )
    return _tile_sides(left_cells, right_cells)


def stack_rgb_marker_mosaic(
    left_rgb: list[np.ndarray],
    right_rgb: list[np.ndarray],
    left_marker: list[np.ndarray],
    right_marker: list[np.ndarray],
    frame_index: int,
    *,
    out_size: int = 224,
    n: int = HISTORY_FRAMES,
    stride: int = HISTORY_STRIDE,
) -> np.ndarray:
    """RGB 4x4 on top, marker 4x4 below; each cell ``INTER_AREA`` then tiled.

    Output is square ``out_size``. Cells are ``out_size/8`` tall and
    ``out_size/4`` wide so the stacked 8x4 grid fills the square without
    resampling across the left|right or RGB|marker seams.
    """
    size = int(out_size)
    cell_h, cell_w = stacked_mosaic_cell_hw(size)
    rgb_left, rgb_right = _history_cells(
        left_rgb, right_rgb, frame_index, n=n, stride=stride
    )
    marker_left, marker_right = _history_cells(
        left_marker, right_marker, frame_index, n=n, stride=stride
    )
    rgb_left = [resize_bgr(frame, cell_w, cell_h) for frame in rgb_left]
    rgb_right = [resize_bgr(frame, cell_w, cell_h) for frame in rgb_right]
    marker_left = [resize_bgr(frame, cell_w, cell_h) for frame in marker_left]
    marker_right = [resize_bgr(frame, cell_w, cell_h) for frame in marker_right]
    mosaic = np.empty((size, size, 3), dtype=np.uint8)
    _tile_sides(rgb_left, rgb_right, mosaic[: size // 2])
    _tile_sides(marker_left, marker_right, mosaic[size // 2 :])
    return mosaic


def resize_history_mosaic(
    left_frames: list[np.ndarray],
    right_frames: list[np.ndarray],
    frame_index: int,
    *,
    out_size: int = 224,
    n: int = HISTORY_FRAMES,
    stride: int = HISTORY_STRIDE,
) -> np.ndarray:
    """Per-cell ``INTER_AREA`` resize, then a seamless 4x4 square mosaic.

    Native 240x320 gels become 56x56 cells at ``out_size=224``. Resizing each
    cell alone avoids resampling across the left|right seam and the raster
    time-wrap between rows.
    """
    cell = mosaic_cell_size(out_size)
    left_cells, right_cells = _history_cells(
        left_frames, right_frames, frame_index, n=n, stride=stride
    )
    left_cells = [resize_bgr(frame, cell, cell) for frame in left_cells]
    right_cells = [resize_bgr(frame, cell, cell) for frame in right_cells]
    return _tile_sides(left_cells, right_cells)


def rest_frame(frames: list[np.ndarray], *, n: int = REST_FRAMES) -> np.ndarray:
    """Median of the first ``n`` frames, used as an undeformed gel reference."""
    if not frames:
        raise ValueError("frames must be non-empty")
    count = max(1, min(int(n), len(frames)))
    stack = np.stack([_bgr_uint8(frame) for frame in frames[:count]], axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


def dense_optical_flow(
    previous: np.ndarray,
    current: np.ndarray,
) -> np.ndarray:
    """Farneback flow from ``previous`` to ``current`` BGR frames."""
    prev_gray = cv2.cvtColor(_bgr_uint8(previous), cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(_bgr_uint8(current), cv2.COLOR_BGR2GRAY)
    return cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )


def _bgr_to_raft_tensor(frame: np.ndarray) -> Any:
    import torch

    rgb = _bgr_uint8(frame)[:, :, ::-1]
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
    return tensor


def raft_optical_flow(
    previous: np.ndarray,
    current: np.ndarray,
    *,
    model: Any,
    device: Any,
) -> np.ndarray:
    """RAFT flow from ``previous`` to ``current``, matching MMFlow's pair-wise demo."""
    import torch
    import torch.nn.functional as functional

    image1 = _bgr_to_raft_tensor(previous).float().unsqueeze(0) / 255.0
    image2 = _bgr_to_raft_tensor(current).float().unsqueeze(0) / 255.0
    image1 = image1 * 2.0 - 1.0
    image2 = image2 * 2.0 - 1.0
    _, _, height, width = image1.shape
    pad_h = (8 - height % 8) % 8
    pad_w = (8 - width % 8) % 8
    if pad_h or pad_w:
        image1 = functional.pad(image1, (0, pad_w, 0, pad_h))
        image2 = functional.pad(image2, (0, pad_w, 0, pad_h))
    image1 = image1.to(device)
    image2 = image2.to(device)
    with torch.inference_mode():
        predictions = model(image1, image2)
    flow = predictions[-1][0, :, :height, :width].permute(1, 2, 0).detach().cpu().numpy()
    return np.ascontiguousarray(flow.astype(np.float32))


def load_raft_small(device: str | None = None) -> tuple[Any, Any]:
    """Load torchvision RAFT-small. Lazy so CPU tests do not download weights."""
    import torch
    from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    weights = Raft_Small_Weights.DEFAULT
    model = raft_small(weights=weights)
    model.eval()
    model.to(device)
    return model, device


def make_color_wheel(bins: tuple[int, ...] | None = None) -> np.ndarray:
    """Middlebury color wheel used by MMFlow / ``mmcv.flow2rgb``."""
    if bins is None:
        bins = (15, 6, 4, 11, 13, 6)
    if len(bins) != 6:
        raise ValueError(f"expected 6 color ranges, got {bins}")
    red_yellow, yellow_green, green_cyan, cyan_blue, blue_magenta, magenta_red = bins
    ranges = [
        [1, np.arange(red_yellow) / red_yellow, 0],
        [1 - np.arange(yellow_green) / yellow_green, 1, 0],
        [0, 1, np.arange(green_cyan) / green_cyan],
        [0, 1 - np.arange(cyan_blue) / cyan_blue, 1],
        [np.arange(blue_magenta) / blue_magenta, 0, 1],
        [1, 0, 1 - np.arange(magenta_red) / magenta_red],
    ]
    num_bins = int(sum(bins))
    color_wheel = np.zeros((3, num_bins), dtype=np.float32)
    col = 0
    for width, color in zip(bins, ranges):
        for channel in range(3):
            color_wheel[channel, col : col + width] = color[channel]
        col += width
    return color_wheel.T


def flow2rgb(flow: np.ndarray, *, unknown_thr: float = 1.0e6) -> np.ndarray:
    """Convert ``(H, W, 2)`` flow to an RGB float image in ``[0, 1]``.

    Matches MMFlow ``visualize_flow`` / ``mmcv.flow2rgb``: scale by the map's
    max radius, map direction with the Middlebury wheel, and render small
    vectors as lighter colors (zero flow is white).
    """
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(f"Expected HxWx2 flow, got {flow.shape}")
    color_wheel = make_color_wheel()
    num_bins = color_wheel.shape[0]
    dx = np.asarray(flow[:, :, 0], dtype=np.float32).copy()
    dy = np.asarray(flow[:, :, 1], dtype=np.float32).copy()
    ignore = np.isnan(dx) | np.isnan(dy) | (np.abs(dx) > unknown_thr) | (np.abs(dy) > unknown_thr)
    dx[ignore] = 0
    dy[ignore] = 0
    radius = np.sqrt(dx * dx + dy * dy)
    max_radius = float(np.max(radius)) if radius.size else 0.0
    if max_radius < 1.0e-3:
        image = np.ones((*dx.shape, 3), dtype=np.float32)
        image[ignore, :] = 0
        return image
    dx = dx / max_radius
    dy = dy / max_radius
    radius = np.sqrt(dx * dx + dy * dy)
    angle = np.arctan2(-dy, -dx) / np.pi
    bin_real = (angle + 1.0) / 2.0 * (num_bins - 1)
    bin_left = np.floor(bin_real).astype(np.int32)
    bin_right = (bin_left + 1) % num_bins
    weight = (bin_real - bin_left.astype(np.float32))[..., None]
    image = (1.0 - weight) * color_wheel[bin_left] + weight * color_wheel[bin_right]
    small = radius <= 1.0
    image[small] = 1.0 - radius[small, None] * (1.0 - image[small])
    image[~small] *= 0.75
    image[ignore, :] = 0
    return image


def colorize_flow(
    flow: np.ndarray,
    *,
    mag_clip: float | None = None,
) -> np.ndarray:
    """MMFlow-style Middlebury map as BGR uint8 for OpenCV / ffmpeg.

    ``mag_clip`` is unused; MMFlow scales each map by its own max radius.
    """
    del mag_clip
    rgb = np.clip(flow2rgb(flow) * 255.0, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb[:, :, ::-1])


def marker_centers(frame: np.ndarray) -> np.ndarray:
    """Detect dark gel-marker locations in a BGR or gray frame, shape ``(N, 2)``."""
    image = np.asarray(frame)
    if image.ndim == 3:
        gray = cv2.cvtColor(_bgr_uint8(image), cv2.COLOR_BGR2GRAY)
    else:
        gray = np.ascontiguousarray(image)
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
    inverted = cv2.GaussianBlur(255 - gray, (5, 5), 0)
    points = cv2.goodFeaturesToTrack(
        inverted,
        maxCorners=150,
        qualityLevel=0.03,
        minDistance=10,
        blockSize=5,
    )
    if points is None:
        return np.zeros((0, 2), dtype=np.float32)
    return points.reshape(-1, 2).astype(np.float32)


def densify_marker_flow(
    flow: np.ndarray,
    points: np.ndarray,
    *,
    sigma: float = MARKER_SPLAT_SIGMA,
    mag_max: float = FLOW_OUTLIER_PX,
) -> np.ndarray:
    """Sample Farneback at markers and splat a dense Middlebury-ready field."""
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(f"Expected HxWx2 flow, got {flow.shape}")
    height, width = flow.shape[:2]
    acc_u = np.zeros((height, width), dtype=np.float32)
    acc_v = np.zeros((height, width), dtype=np.float32)
    acc_w = np.zeros((height, width), dtype=np.float32)
    max_sq = float(mag_max) * float(mag_max)
    for x_raw, y_raw in np.asarray(points, dtype=np.float32).reshape(-1, 2):
        col = int(round(float(x_raw)))
        row = int(round(float(y_raw)))
        if not (0 <= col < width and 0 <= row < height):
            continue
        delta_u = float(flow[row, col, 0])
        delta_v = float(flow[row, col, 1])
        if delta_u * delta_u + delta_v * delta_v > max_sq:
            continue
        acc_u[row, col] = delta_u
        acc_v[row, col] = delta_v
        acc_w[row, col] = 1.0
    kernel = int(float(sigma) * 6.0) | 1
    acc_u = cv2.GaussianBlur(acc_u, (kernel, kernel), float(sigma))
    acc_v = cv2.GaussianBlur(acc_v, (kernel, kernel), float(sigma))
    acc_w = cv2.GaussianBlur(acc_w, (kernel, kernel), float(sigma))
    dense = np.zeros((height, width, 2), dtype=np.float32)
    valid = acc_w > 1.0e-4
    dense[:, :, 0][valid] = acc_u[valid] / acc_w[valid]
    dense[:, :, 1][valid] = acc_v[valid] / acc_w[valid]
    return dense


def consecutive_flow_frames(
    frames: list[np.ndarray],
    *,
    stride: int = 1,
    mag_clip: float = CAMERA_FLOW_MAG_CLIP,
    method: str = "farneback",
    raft_model: Any | None = None,
    raft_device: Any | None = None,
) -> list[np.ndarray]:
    """Pair-wise flow then MMFlow ``flow2rgb``, as in ``demo/video_demo.py``.

    ``method='raft'`` uses torchvision RAFT-small (MMFlow-style learned flow).
    ``method='farneback'`` stays CPU-only. Frame 0 is zero-flow (white).
    """
    del mag_clip
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    method = str(method).lower()
    if method not in {"farneback", "raft"}:
        raise ValueError(f"unknown flow method {method}")
    model = raft_model
    device = raft_device
    if method == "raft" and model is None:
        model, device = load_raft_small()
    colored: list[np.ndarray] = []
    for index, frame in enumerate(frames):
        previous = frames[max(0, index - int(stride))]
        if method == "raft":
            flow = raft_optical_flow(previous, frame, model=model, device=device)
        else:
            flow = dense_optical_flow(previous, frame)
        colored.append(colorize_flow(flow))
    return colored


def flow_history_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Colorize dense optical flow from the undeformed rest gel to each frame.

    Adjacent GelSight frames are quasi-static, so a 5-frame gap is near zero.
    The useful field is Farneback(rest → t), sampled on markers and splatted
    into MMFlow's Middlebury ``flow2rgb`` map. The 4x4 mosaic still uses stride 5.
    """
    reference = rest_frame(frames)
    points = marker_centers(reference)
    colored: list[np.ndarray] = []
    for frame in frames:
        flow = dense_optical_flow(reference, frame)
        if len(points) > 0:
            flow = densify_marker_flow(flow, points)
        colored.append(colorize_flow(flow))
    return colored


def _ffmpeg_video_shape(path: Path) -> tuple[int, int]:
    """Return ``(width, height)`` for the first video stream using ffprobe."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise FileNotFoundError("ffprobe is required for the video decoder fallback")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            str(path),
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    value = result.stdout.strip()
    try:
        width, height = (int(part) for part in value.split("x", 1))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"ffprobe returned an invalid video shape for {path}: {value!r}") from exc
    if width <= 0 or height <= 0:
        raise RuntimeError(f"ffprobe returned a non-positive video shape for {path}: {value!r}")
    return width, height


def _read_exact(stream: Any, size: int) -> bytes:
    """Read one complete raw-video frame from a pipe, including short pipe reads."""
    chunks: list[bytes] = []
    remaining = int(size)
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _decode_video_with_ffmpeg(path: Path) -> list[np.ndarray]:
    """Decode frames through FFmpeg when OpenCV cannot decode a video stream.

    Some OpenCV builds select an unavailable hardware AV1 decoder and return no
    frames even though the same file is decodable by FFmpeg's software decoder.
    ``libdav1d`` is selected explicitly for AV1 when it is available; other
    codecs use FFmpeg's normal software decoder selection.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return []
    width, height = _ffmpeg_video_shape(path)
    process = subprocess.Popen(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    frame_bytes = width * height * 3
    frames: list[np.ndarray] = []
    try:
        if process.stdout is None:
            return []
        while True:
            raw = _read_exact(process.stdout, frame_bytes)
            if not raw:
                break
            if len(raw) != frame_bytes:
                raise RuntimeError(f"FFmpeg returned a partial frame for {path}")
            frames.append(np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy())
        return_code = process.wait()
    finally:
        if process.stdout is not None:
            process.stdout.close()
        if process.poll() is None:
            process.kill()
            process.wait()
    if return_code != 0:
        raise RuntimeError(f"FFmpeg failed to decode {path} with exit code {return_code}")
    return frames


def decode_video(path: Path) -> list[np.ndarray]:
    """Decode an entire mp4 to BGR frames, with an FFmpeg AV1 fallback."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video {path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(_bgr_uint8(frame))
    capture.release()
    if frames:
        return frames
    try:
        frames = _decode_video_with_ffmpeg(path)
    except (FileNotFoundError, RuntimeError, subprocess.SubprocessError) as exc:
        message = f"no frames decoded from {path}; FFmpeg fallback failed: {exc}"
        raise RuntimeError(message) from exc
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return frames


def episode_slice(
    file_frames: list[np.ndarray],
    *,
    from_timestamp: float,
    to_timestamp: float,
    length: int,
    fps: float = DEFAULT_FPS,
) -> list[np.ndarray]:
    """Slice one LeRobot episode from a concatenated video file."""
    start = int(round(float(from_timestamp) * float(fps)))
    end = int(round(float(to_timestamp) * float(fps)))
    start = max(0, min(start, len(file_frames)))
    end = max(start, min(end, len(file_frames)))
    sliced = file_frames[start:end]
    if length > 0 and len(sliced) != int(length):
        if abs(len(sliced) - int(length)) <= 1:
            if len(sliced) > int(length):
                sliced = sliced[: int(length)]
            elif sliced:
                sliced = sliced + [sliced[-1]] * (int(length) - len(sliced))
    if not sliced:
        raise RuntimeError(
            f"empty episode slice start={start} end={end} file_len={len(file_frames)}"
        )
    return sliced


def pack_episodes_into_videos(
    lengths: list[int],
    videos: list[tuple[int, int, float]],
    *,
    fps: float = DEFAULT_FPS,
) -> list[tuple[int, int, float, float]]:
    """Assign each episode to a concatenated LeRobot v3 video file.

    ``videos`` is ``(chunk_index, file_index, duration_s)`` in file order.
    Returns ``(chunk_index, file_index, from_s, to_s)`` per episode.
    """
    if not videos:
        raise ValueError("at least one video file is required")
    packed: list[tuple[int, int, float, float]] = []
    video_index = 0
    cursor = 0.0
    chunk, file_index, duration = videos[0]
    rate = float(fps)
    for length in lengths:
        need = int(length) / rate
        while cursor + need > duration + 1.0e-2 and video_index + 1 < len(videos):
            video_index += 1
            cursor = 0.0
            chunk, file_index, duration = videos[video_index]
        packed.append((int(chunk), int(file_index), float(cursor), float(cursor + need)))
        cursor += need
    return packed


def _ffprobe_duration_s(path: Path) -> float:
    import subprocess

    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        text=True,
    ).strip()
    return float(out)


def episode_resume_index_fields(
    episodes: list[tuple[int, int, int, int]],
    *,
    instruction: str,
) -> dict[str, list[Any]]:
    """LeRobot v3 episode-index columns required to resume recording.

    ``episodes`` is ``(episode_index, length, data_chunk, data_file)`` sorted by
    episode index. Resume uses ``dataset_to_index`` plus the last data chunk/file
    to append the next parquet shard; a truncated rebuild that only stores
    ``episode_index`` and ``length`` raises ``KeyError('dataset_to_index')``.
    """
    from_index = 0
    indices: list[int] = []
    lengths: list[int] = []
    chunks: list[int] = []
    files: list[int] = []
    from_indices: list[int] = []
    to_indices: list[int] = []
    tasks: list[list[str]] = []
    meta_chunk: list[int] = []
    meta_file: list[int] = []
    text = str(instruction or "")
    for episode_index, length, chunk, file_index in episodes:
        start = from_index
        stop = start + int(length)
        from_index = stop
        indices.append(int(episode_index))
        lengths.append(int(length))
        chunks.append(int(chunk))
        files.append(int(file_index))
        from_indices.append(int(start))
        to_indices.append(int(stop))
        tasks.append([text])
        meta_chunk.append(0)
        meta_file.append(0)
    return {
        "episode_index": indices,
        "length": lengths,
        "tasks": tasks,
        "data/chunk_index": chunks,
        "data/file_index": files,
        "dataset_from_index": from_indices,
        "dataset_to_index": to_indices,
        "meta/episodes/chunk_index": meta_chunk,
        "meta/episodes/file_index": meta_file,
    }


def _parse_chunk_file_indices(path: Path) -> tuple[int, int]:
    chunk = int(path.parent.name.split("-")[1])
    file_index = int(path.stem.split("-")[1])
    return chunk, file_index


def _dataset_instruction(root: Path) -> str:
    manifest = root / "tacbenchhub_expert_manifest.json"
    if manifest.is_file():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
            payload = None
        if isinstance(payload, dict):
            text = str(payload.get("instruction") or "").strip()
            if text:
                return text
    tasks_path = root / "meta" / "tasks.parquet"
    if tasks_path.is_file():
        import pandas as pd

        try:
            frame = pd.read_parquet(tasks_path)
        except (OSError, ValueError, TypeError):
            frame = None
        if frame is not None and len(frame.index):
            label = str(frame.index[0]).strip()
            if label and label.lower() != "task_index":
                return label
            if "task" in frame.columns:
                return str(frame["task"].iloc[0]).strip()
    return ""


def rebuild_truncated_episode_index(dataset_root: Path) -> int:
    """Rewrite ``meta/episodes`` from per-episode data parquet + video durations.

    Some 20260823 insertion roots kept 100 data files but only episode 99 in
    the episode index. Video packing matches healthy LeRobot v3 roots.
    Resume also needs ``dataset_to_index`` and data shard coordinates.
    """
    import pandas as pd

    root = Path(dataset_root)
    data_files = sorted((root / "data").rglob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"no data parquet under {root / 'data'}")
    episodes: list[tuple[int, int, int, int]] = []
    for path in data_files:
        frame = pd.read_parquet(path, columns=["episode_index"])
        chunk, file_index = _parse_chunk_file_indices(path)
        episodes.append(
            (int(frame["episode_index"].iloc[0]), int(len(frame)), chunk, file_index)
        )
    episodes.sort(key=lambda item: item[0])
    lengths = [length for _index, length, _chunk, _file in episodes]
    video_root = root / "videos"
    if not video_root.is_dir():
        raise FileNotFoundError(f"no videos directory under {root}")
    columns: dict[str, list[Any]] = episode_resume_index_fields(
        episodes,
        instruction=_dataset_instruction(root),
    )
    for key_dir in sorted(p for p in video_root.iterdir() if p.is_dir()):
        key = key_dir.name
        files = sorted(key_dir.rglob("*.mp4"))
        if not files:
            continue
        videos: list[tuple[int, int, float]] = []
        for path in files:
            chunk = int(path.parent.name.split("-")[1])
            file_index = int(path.stem.split("-")[1])
            videos.append((chunk, file_index, _ffprobe_duration_s(path)))
        packed = pack_episodes_into_videos(lengths, videos)
        prefix = f"videos/{key}"
        columns[f"{prefix}/chunk_index"] = [row[0] for row in packed]
        columns[f"{prefix}/file_index"] = [row[1] for row in packed]
        columns[f"{prefix}/from_timestamp"] = [row[2] for row in packed]
        columns[f"{prefix}/to_timestamp"] = [row[3] for row in packed]
    out_dir = root / "meta" / "episodes" / "chunk-000"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "file-000.parquet"
    backup = out_dir / "file-000.parquet.truncated"
    if dest.is_file() and not backup.is_file():
        dest.replace(backup)
    pd.DataFrame(columns).to_parquet(dest, index=False)
    return len(episodes)


def _video_coords(row: Any, key: str) -> dict[str, Any]:
    prefix = f"videos/{key}"
    return {
        "chunk": int(row[f"{prefix}/chunk_index"]),
        "file": int(row[f"{prefix}/file_index"]),
        "from": float(row[f"{prefix}/from_timestamp"]),
        "to": float(row[f"{prefix}/to_timestamp"]),
    }


def iter_episode_rows(dataset_root: Path) -> list[dict[str, Any]]:
    """Load LeRobot v3 episode rows with tactile and marker video coordinates."""
    import pandas as pd

    episode_dir = dataset_root / "meta" / "episodes"
    paths = sorted(episode_dir.glob("chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no episode parquet under {episode_dir}")
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        item: dict[str, Any] = {
            "episode_index": int(row["episode_index"]),
            "length": int(row["length"]),
            "tactile_left": _video_coords(row, "observation.images.tactile_left"),
            "tactile_right": _video_coords(row, "observation.images.tactile_right"),
        }
        marker_left_key = "videos/observation.images.marker_left/chunk_index"
        if marker_left_key in row.index:
            item["marker_left"] = _video_coords(row, "observation.images.marker_left")
            item["marker_right"] = _video_coords(row, "observation.images.marker_right")
        agentview_key = "videos/observation.images.agentview/chunk_index"
        if agentview_key in row.index:
            item["agentview"] = _video_coords(row, "observation.images.agentview")
        wrist_key = "videos/observation.images.wrist/chunk_index"
        if wrist_key in row.index:
            item["wrist"] = _video_coords(row, "observation.images.wrist")
        rows.append(item)
    return rows


def capped_episode_rows(
    dataset_root: Path, max_episodes: int | None = None
) -> list[dict[str, Any]]:
    """Episode rows sorted by ``episode_index``, optionally truncated."""
    rows = sorted(
        iter_episode_rows(dataset_root),
        key=lambda row: int(row["episode_index"]),
    )
    if max_episodes is None:
        return rows
    return rows[: max(0, int(max_episodes))]
