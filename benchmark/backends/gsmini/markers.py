"""Shared FOTS marker rendering for recorded and live GSmini observations.

Drawing adapted from TacEx/Tabero; Apache-2.0, see export.py for provenance.
"""

from functools import lru_cache
import numpy as np


@lru_cache(maxsize=1)
def _marker_patches():
    """TacEx's subpixel soft-circle templates for its default 3-pixel radius.

    Retain the original 10x supersampling, blur and cubic downsampling; only
    cache the default radius instead of generating all 50 unused size slots.
    """
    import cv2

    patches = np.empty((10, 10, 12, 12), dtype=np.uint8)
    for u in range(10):
        for v in range(10):
            patch = np.full((120, 120), 255, dtype=np.uint8)
            cv2.circle(
                patch,
                (60 + u, 60 + v),
                30,
                (0, 0, 0),
                thickness=cv2.FILLED,
                lineType=cv2.LINE_AA,
            )
            patch = cv2.GaussianBlur(patch, (17, 17), 15)
            patches[u, v] = cv2.resize(patch, (12, 12), interpolation=cv2.INTER_CUBIC)
    return patches


def _draw_markers(image, vectors, motion):
    """Overlay dots and, when requested, displacement arrows in place."""
    _draw_dots(image, motion)
    if vectors is not None:
        _draw_deformation(vectors, motion)


def _draw_dots(image, motion):
    _, current = motion
    height, width = image.shape[:2]
    dots = np.full((height + 24, width + 24), 255, dtype=np.uint8)
    patches = _marker_patches()
    centers = current + np.array([0.5, 0.5]) + 12
    floors = np.floor(centers).astype(int)
    phases = np.floor((centers - floors) * 10).astype(int)
    for (floor_u, floor_v), (patch_u, patch_v) in zip(floors, phases):
        x, y = floor_u - 6, floor_v - 6
        if 0 <= x < dots.shape[1] - 12 and 0 <= y < dots.shape[0] - 12:
            dots[y : y + 12, x : x + 12] = patches[patch_u, patch_v]
    np.multiply(
        image,
        dots[12:-12, 12:-12, None].astype(np.float64) / 255,
        out=image,
        casting="unsafe",
    )


def _draw_deformation(vectors, motion):
    import cv2

    initial, current = motion
    height, width = vectors.shape[:2]
    arrows = np.ones((height, width), dtype=np.uint8)
    # The upstream renderer truncates coordinates before drawing; arrow_scale=1 adds one
    # displacement beyond the current point, starting at the initial point.
    for start, point in zip(initial.astype(int), current.astype(int)):
        x, y = point
        if not (0 <= x < width and 0 <= y < height):
            continue
        cv2.arrowedLine(
            arrows,
            tuple(start),
            tuple(point + (point - start)),
            (0, 0, 0),
            2,
            tipLength=0.2,
        )
    arrows = cv2.normalize(
        arrows, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    np.multiply(
        vectors,
        arrows[..., None].astype(np.float64) / 255,
        out=vectors,
        casting="unsafe",
    )


def marker_image(rgb, motion):
    """Overlay the current subpixel dots on an RGB uint8 sensor frame."""
    return _render_images(rgb, motion, deformation=False)[0]


def deformation_image(rgb, motion):
    """Overlay displacement arrows using the exported Deformation view's pixels."""
    vectors = np.asarray(rgb).copy()
    _draw_deformation(vectors, np.asarray(motion))
    return vectors


def marker_images(rgb, motion):
    """Return separate dot-only Marker and arrow-only Deformation RGB views."""
    return _render_images(rgb, motion, deformation=True)


def _render_images(rgb, motion, *, deformation):
    image = np.asarray(rgb)
    motion = np.asarray(motion)
    result = image.copy()
    vectors = image.copy() if deformation else None
    _draw_markers(result, vectors, motion)
    return result, vectors
