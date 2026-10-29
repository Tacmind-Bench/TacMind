"""Opt-in evaluation looks migrated from TacBenchHub; no simulator imports at load time.

The preset catalog, CLI expansion and orbit math follow
TacBenchHub/tacbenchhub/evaluation/perturbations.py. Runtime adapters in learning
use TacMind's own task/reset, lighting and recording contracts.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Literal

PERTURBATION_KINDS = ("lighting", "camera", "scale")
DEFAULT_LIGHTING_PRESET = "warm"
CAMERA_DISTANCE_SCALES = {
    "far_110": 1.10,
    "far_120": 1.20,
}
CAMERA_PRESET_NAMES = ("left", "right", "high", "low", *CAMERA_DISTANCE_SCALES)
YAW_CAMERA_PRESETS = frozenset({"left", "right"})
PITCH_CAMERA_PRESETS = frozenset({"high", "low"})
DISTANCE_CAMERA_PRESETS = frozenset(CAMERA_DISTANCE_SCALES)
DEFAULT_CAMERA_SIDE: Literal["left", "right"] = "left"
DEFAULT_CAMERA_PRESET = "left"
DEFAULT_CAMERA_YAW_DEG = 45.0
DEFAULT_CAMERA_PITCH_DEG = 10.0
MIN_EYE_HEIGHT_ABOVE_LOOK_AT_M = 0.08
DEFAULT_OBJECT_SCALE = 0.90
SWEEP_OBJECT_SCALES = (0.85, 0.90)
WORKSPACE_HALF_XY_M = 0.45
WORKSPACE_HEIGHT_M = 0.70
FAR_CLIP_MARGIN = 1.2
MAX_ORBIT_RADIUS_SCALE = 2.5


@dataclass(frozen=True)
class LightingLook:
    """One physically motivated lighting condition.

    ``intensity_scale`` applies to direct/local lights and
    ``dome_intensity_scale`` applies to environment fill. ``color`` is an
    approximate blackbody RGB for direct/local lights; ``dome_color`` is the
    milder fill tint multiplied onto the DomeLight (and therefore the visible
    HDRI). ``color_temperature`` records the corresponding physical CCT.
    ``key_placement="behind_subject"`` moves local (non-dome) lights behind the
    workspace, opposite AgentView. That is ``backlight`` / 逆光, not a side key.
    Built-in looks do not write a frame-wide RTX color grade. Night dims USD
    lights with ``exposure_ev``; official histogram auto-exposure stays on.
    """

    name: str
    color: tuple[float, float, float] | None = None
    intensity_scale: float = 1.0
    dome_color: tuple[float, float, float] | None = None
    dome_intensity_scale: float | None = None
    color_temperature: float | None = None
    key_placement: str = "authored"
    key_radius: float | None = None
    exposure_ev: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "color": None if self.color is None else list(self.color),
            "dome_color": None if self.dome_color is None else list(self.dome_color),
            "color_temperature": (
                None
                if self.color_temperature is None
                else float(self.color_temperature)
            ),
            "intensity_scale": float(self.intensity_scale),
            "dome_intensity_scale": (
                None
                if self.dome_intensity_scale is None
                else float(self.dome_intensity_scale)
            ),
            "key_placement": self.key_placement,
            "key_radius": None if self.key_radius is None else float(self.key_radius),
            "exposure_ev": float(self.exposure_ev),
        }


LIGHTING_LOOKS: dict[str, LightingLook] = {
    "warm": LightingLook(
        "warm",
        color=(1.0, 0.48, 0.16),
        dome_color=(1.0, 0.52, 0.18),
        intensity_scale=1.25,
        dome_intensity_scale=0.55,
        color_temperature=2700.0,
    ),
    "cool": LightingLook(
        "cool",
        color=(0.48, 0.68, 1.0),
        dome_color=(0.42, 0.64, 1.0),
        intensity_scale=1.20,
        dome_intensity_scale=0.95,
        color_temperature=7500.0,
    ),
    "fluorescent": LightingLook(
        "fluorescent",
        color=(0.72, 1.0, 0.62),
        dome_color=(0.78, 1.0, 0.68),
        intensity_scale=1.05,
        dome_intensity_scale=0.85,
        color_temperature=4000.0,
    ),
    "dusk": LightingLook(
        "dusk",
        color=(1.0, 0.22, 0.05),
        dome_color=(1.0, 0.18, 0.03),
        intensity_scale=0.42,
        dome_intensity_scale=0.22,
        color_temperature=2000.0,
    ),
    "overcast": LightingLook(
        "overcast",
        color=(0.62, 0.76, 1.0),
        dome_color=(0.55, 0.72, 1.0),
        intensity_scale=0.28,
        dome_intensity_scale=1.60,
        color_temperature=7000.0,
    ),
    "night": LightingLook(
        "night",
        color=(1.0, 0.96, 0.90),
        dome_color=(0.96, 0.94, 0.90),
        intensity_scale=0.035,
        dome_intensity_scale=0.012,
        color_temperature=4000.0,
        exposure_ev=1.5,
    ),
    # 逆光: small hard key behind the subject, dim cool fill. Not a side key.
    "backlight": LightingLook(
        "backlight",
        color=(1.0, 0.96, 0.88),
        dome_color=(0.40, 0.52, 1.0),
        intensity_scale=3.50,
        dome_intensity_scale=0.07,
        color_temperature=5600.0,
        key_placement="behind_subject",
        key_radius=0.35,
    ),
}
DEFAULT_LIGHTING_PRESET_NAMES = (
    "warm",
    "cool",
    "fluorescent",
    "night",
)


@dataclass(frozen=True)
class AgentviewOrbitResult:
    """Resolved agentview extrinsics after an orbit around the look-at point."""

    position: tuple[float, float, float]
    rotation_wxyz: tuple[float, float, float, float]
    look_at: tuple[float, float, float]
    yaw_rad: float
    clipping_range: tuple[float, float]
    radius_scale: float
    pitch_rad: float = 0.0
    distance_scale: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": list(self.position),
            "rotation_wxyz": list(self.rotation_wxyz),
            "look_at": list(self.look_at),
            "yaw_deg": math.degrees(self.yaw_rad),
            "pitch_deg": math.degrees(self.pitch_rad),
            "clipping_range": list(self.clipping_range),
            "radius_scale": self.radius_scale,
            "distance_scale": self.distance_scale,
        }


@dataclass(frozen=True)
class EvalPerturbationSpec:
    """Stackable eval-only robustness knobs. Identity means official eval."""

    lighting: bool = False
    camera: bool = False
    scale: bool = False
    lighting_preset: str = DEFAULT_LIGHTING_PRESET
    camera_preset: str = DEFAULT_CAMERA_PRESET
    camera_yaw_deg: float = DEFAULT_CAMERA_YAW_DEG
    camera_side: str = DEFAULT_CAMERA_SIDE
    camera_pitch_deg: float = DEFAULT_CAMERA_PITCH_DEG
    camera_radius_scale: float = 1.0
    object_scale: float = DEFAULT_OBJECT_SCALE


    @property
    def enabled(self) -> bool:
        return bool(self.lighting or self.camera or self.scale)

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(kind for kind in PERTURBATION_KINDS if getattr(self, kind))

    @property
    def effective_object_scale(self) -> float:
        return float(self.object_scale) if self.scale else 1.0

    @property
    def signed_camera_yaw_rad(self) -> float:
        if not self.camera or self.camera_preset not in YAW_CAMERA_PRESETS:
            return 0.0
        sign = 1.0 if self.camera_side == "left" else -1.0
        return sign * math.radians(float(self.camera_yaw_deg))

    @property
    def signed_camera_pitch_rad(self) -> float:
        if not self.camera or self.camera_preset not in PITCH_CAMERA_PRESETS:
            return 0.0
        sign = 1.0 if self.camera_preset == "high" else -1.0
        return sign * math.radians(float(self.camera_pitch_deg))

    @property
    def camera_distance_scale(self) -> float:
        if not self.camera or self.camera_preset not in DISTANCE_CAMERA_PRESETS:
            return 1.0
        return float(self.camera_radius_scale)

    def lighting_look(self) -> LightingLook | None:
        if not self.lighting:
            return None
        return LIGHTING_LOOKS[self.lighting_preset]

    def to_dict(self) -> dict[str, Any]:
        look = self.lighting_look()
        camera = bool(self.camera)
        yaw_look = camera and self.camera_preset in YAW_CAMERA_PRESETS
        pitch_look = camera and self.camera_preset in PITCH_CAMERA_PRESETS
        distance_look = camera and self.camera_preset in DISTANCE_CAMERA_PRESETS
        return {
            "enabled": self.enabled,
            "kinds": list(self.kinds),
            "lighting_preset": None if look is None else look.name,
            "lighting": None if look is None else look.to_dict(),
            "camera_preset": self.camera_preset if camera else None,
            "camera_side": self.camera_side if yaw_look else None,
            "camera_yaw_deg": float(self.camera_yaw_deg) if yaw_look else None,
            "camera_pitch_deg": float(self.camera_pitch_deg) if pitch_look else None,
            "camera_radius_scale": (
                float(self.camera_radius_scale) if distance_look else None
            ),
            "object_scale": self.effective_object_scale,
        }


def catalog_robustness_looks() -> tuple[tuple[str, EvalPerturbationSpec], ...]:
    """The twelve independent lighting, camera and scale conditions."""
    looks = [
        (f"lighting_{name}", EvalPerturbationSpec(lighting=True, lighting_preset=name))
        for name in DEFAULT_LIGHTING_PRESET_NAMES
    ]
    looks.extend(
        (f"camera_{name}", EvalPerturbationSpec(
            camera=True, camera_preset=name,
            camera_side=name if name in YAW_CAMERA_PRESETS else DEFAULT_CAMERA_SIDE,
            camera_radius_scale=CAMERA_DISTANCE_SCALES.get(name, 1.0)))
        for name in CAMERA_PRESET_NAMES
    )
    looks.extend(
        (f"scale_{round(scale * 100):03d}", EvalPerturbationSpec(scale=True, object_scale=scale))
        for scale in SWEEP_OBJECT_SCALES
    )
    return tuple(looks)


def expand_eval_perturbations(kinds: Sequence[str] | None) -> tuple[tuple[str, EvalPerturbationSpec], ...]:
    """Omitted kinds select nominal conditions; an empty list selects all presets."""
    if kinds is None:
        return (("identity", EvalPerturbationSpec()),)
    selected = set(kinds or PERTURBATION_KINDS)
    return tuple((name, spec) for name, spec in catalog_robustness_looks()
                 if selected.intersection(spec.kinds))


def backlight_key_position(
    camera_xyz: Sequence[float],
    look_at_xyz: Sequence[float],
    *,
    distance_m: float = 0.50,
    height_boost_m: float = 1.25,
) -> tuple[float, float, float]:
    """Place a key light behind the look-at, opposite AgentView (逆光)."""
    camera = (float(camera_xyz[0]), float(camera_xyz[1]), float(camera_xyz[2]))
    look_at = (float(look_at_xyz[0]), float(look_at_xyz[1]), float(look_at_xyz[2]))
    dx = look_at[0] - camera[0]
    dy = look_at[1] - camera[1]
    horizontal = math.hypot(dx, dy)
    if horizontal < 1.0e-6:
        dx, dy, horizontal = -1.0, 0.0, 1.0
    return (
        look_at[0] + float(distance_m) * dx / horizontal,
        look_at[1] + float(distance_m) * dy / horizontal,
        look_at[2] + float(height_boost_m),
    )


def horizontal_fov_rad(focal_length: float, horizontal_aperture: float) -> float:
    if focal_length <= 0.0 or horizontal_aperture <= 0.0:
        raise ValueError("focal length and horizontal aperture must be positive")
    return 2.0 * math.atan(horizontal_aperture / (2.0 * focal_length))


def orbit_agentview(
    position: Sequence[float],
    rotation_wxyz: Sequence[float],
    *,
    yaw_rad: float = 0.0,
    pitch_rad: float = 0.0,
    distance_scale: float = 1.0,
    look_at_z: float,
    horizontal_fov_rad: float,
    clipping_range: Sequence[float],
    workspace_half_xy: float = WORKSPACE_HALF_XY_M,
    workspace_height: float = WORKSPACE_HEIGHT_M,
    preserve_minimum_far_clip: float | None = None,
    pull_back: bool = True,
) -> AgentviewOrbitResult:
    """Orbit the OpenGL agentview about the workspace look-at.

    Yaw is a left/right orbit in the XY plane. Pitch raises or lowers the eye
    around the camera's right axis. Distance scales radius along the look
    vector. Desktop yaw orbits may pull back so the table stays in frame;
    pitch and dolly do not.
    """
    eye = _as_vec3(position)
    rotation = _as_quat(rotation_wxyz)
    near_clip, far_clip = (float(clipping_range[0]), float(clipping_range[1]))
    look_at = _look_at_on_plane(eye, rotation, look_at_z)
    yaw = float(yaw_rad)
    pitch = float(pitch_rad)
    distance = float(distance_scale)
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError("distance_scale must be finite and positive")
    if (
        abs(yaw) < 1.0e-12
        and abs(pitch) < 1.0e-12
        and abs(distance - 1.0) < 1.0e-12
    ):
        return AgentviewOrbitResult(
            position=eye,
            rotation_wxyz=rotation,
            look_at=look_at,
            yaw_rad=0.0,
            clipping_range=(near_clip, far_clip),
            radius_scale=1.0,
            pitch_rad=0.0,
            distance_scale=1.0,
        )
    orbited_eye = eye
    if abs(yaw) >= 1.0e-12:
        orbited_eye = _rotate_around_z(orbited_eye, look_at, yaw)
    if abs(pitch) >= 1.0e-12:
        orbited_eye = _pitch_around_look_at(orbited_eye, look_at, pitch)
    if abs(distance - 1.0) >= 1.0e-12:
        orbited_eye = _add(
            look_at, _scale_vec(_sub(orbited_eye, look_at), distance)
        )
    min_z = look_at[2] + MIN_EYE_HEIGHT_ABOVE_LOOK_AT_M
    if orbited_eye[2] < min_z:
        orbited_eye = (orbited_eye[0], orbited_eye[1], min_z)
    if pull_back and abs(yaw) >= 1.0e-12:
        pulled_eye, radius_scale = _pull_back_to_frame(
            orbited_eye,
            look_at,
            horizontal_fov_rad=float(horizontal_fov_rad),
            near_clip=near_clip,
            far_clip=far_clip,
            workspace_half_xy=float(workspace_half_xy),
            workspace_height=float(workspace_height),
        )
        new_far = max(
            far_clip,
            FAR_CLIP_MARGIN
            * _max_corner_distance(
                pulled_eye,
                look_at,
                workspace_half_xy=float(workspace_half_xy),
                workspace_height=float(workspace_height),
            ),
        )
    else:
        pulled_eye, radius_scale = orbited_eye, 1.0
        new_far = far_clip
    new_rotation = _look_at_opengl_wxyz(pulled_eye, look_at)
    if preserve_minimum_far_clip is not None:
        new_far = max(new_far, float(preserve_minimum_far_clip))
    return AgentviewOrbitResult(
        position=pulled_eye,
        rotation_wxyz=new_rotation,
        look_at=look_at,
        yaw_rad=yaw,
        clipping_range=(near_clip, new_far),
        radius_scale=radius_scale,
        pitch_rad=pitch,
        distance_scale=distance,
    )


def _as_vec3(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError("expected a 3-vector")
    return (float(values[0]), float(values[1]), float(values[2]))


def _as_quat(values: Sequence[float]) -> tuple[float, float, float, float]:
    if len(values) != 4:
        raise ValueError("expected a wxyz quaternion")
    w, x, y, z = (float(values[0]), float(values[1]), float(values[2]), float(values[3]))
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1.0e-12:
        raise ValueError("camera rotation quaternion must be non-zero")
    return (w / norm, x / norm, y / norm, z / norm)


def _sub(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _add(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _scale_vec(
    values: tuple[float, float, float], factor: float
) -> tuple[float, float, float]:
    return (values[0] * factor, values[1] * factor, values[2] * factor)


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _cross(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _norm(values: tuple[float, float, float]) -> float:
    return math.sqrt(_dot(values, values))


def _normalize(values: tuple[float, float, float]) -> tuple[float, float, float]:
    length = _norm(values)
    if length < 1.0e-12:
        raise ValueError("cannot normalize a near-zero vector")
    return _scale_vec(values, 1.0 / length)


def quat_rotate(
    quat_wxyz: Sequence[float], vector: Sequence[float]
) -> tuple[float, float, float]:
    """Rotate a world vector by a wxyz quaternion."""
    w, x, y, z = _as_quat(quat_wxyz)
    vx, vy, vz = _as_vec3(vector)
    uv = _cross((x, y, z), (vx, vy, vz))
    uuv = _cross((x, y, z), uv)
    return (
        vx + 2.0 * w * uv[0] + 2.0 * uuv[0],
        vy + 2.0 * w * uv[1] + 2.0 * uuv[1],
        vz + 2.0 * w * uv[2] + 2.0 * uuv[2],
    )


def _look_at_on_plane(
    eye: tuple[float, float, float],
    rotation_wxyz: tuple[float, float, float, float],
    plane_z: float,
) -> tuple[float, float, float]:
    forward = quat_rotate(rotation_wxyz, (0.0, 0.0, -1.0))
    if abs(forward[2]) < 1.0e-6:
        raise ValueError(
            "camera forward ray is parallel to the requested look-at plane"
        )
    t = (float(plane_z) - eye[2]) / forward[2]
    if t <= 0.0:
        raise ValueError(
            "camera forward ray does not intersect the requested look-at plane "
            "in front of the camera"
        )
    return _add(eye, _scale_vec(forward, t))


def _rotate_around_z(
    eye: tuple[float, float, float],
    look_at: tuple[float, float, float],
    yaw_rad: float,
) -> tuple[float, float, float]:
    rel = _sub(eye, look_at)
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    rotated = (
        cosine * rel[0] - sine * rel[1],
        sine * rel[0] + cosine * rel[1],
        rel[2],
    )
    return _add(look_at, rotated)


def _pitch_around_look_at(
    eye: tuple[float, float, float],
    look_at: tuple[float, float, float],
    pitch_rad: float,
) -> tuple[float, float, float]:
    """Rotate the eye around the look-at. Positive pitch raises the camera."""
    rel = _sub(eye, look_at)
    axis = _cross(rel, (0.0, 0.0, 1.0))
    if _norm(axis) < 1.0e-8:
        axis = (1.0, 0.0, 0.0)
    return _add(look_at, _rotate_rodrigues(rel, axis, pitch_rad))


def _rotate_rodrigues(
    vector: tuple[float, float, float],
    axis: tuple[float, float, float],
    angle_rad: float,
) -> tuple[float, float, float]:
    unit = _normalize(axis)
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return _add(
        _add(_scale_vec(vector, cosine), _scale_vec(_cross(unit, vector), sine)),
        _scale_vec(unit, _dot(unit, vector) * (1.0 - cosine)),
    )


def _workspace_corners(
    look_at: tuple[float, float, float],
    *,
    workspace_half_xy: float,
    workspace_height: float,
) -> tuple[tuple[float, float, float], ...]:
    xs = (look_at[0] - workspace_half_xy, look_at[0] + workspace_half_xy)
    ys = (look_at[1] - workspace_half_xy, look_at[1] + workspace_half_xy)
    zs = (look_at[2], look_at[2] + workspace_height)
    return tuple((x, y, z) for x in xs for y in ys for z in zs)


def _max_corner_distance(
    eye: tuple[float, float, float],
    look_at: tuple[float, float, float],
    *,
    workspace_half_xy: float,
    workspace_height: float,
) -> float:
    return max(
        _norm(_sub(corner, eye))
        for corner in _workspace_corners(
            look_at,
            workspace_half_xy=workspace_half_xy,
            workspace_height=workspace_height,
        )
    )


def _point_in_opengl_frustum(
    point: tuple[float, float, float],
    eye: tuple[float, float, float],
    rotation_wxyz: tuple[float, float, float, float],
    *,
    horizontal_fov_rad: float,
    near_clip: float,
    far_clip: float,
) -> bool:
    delta = _sub(point, eye)
    cam_x = quat_rotate(rotation_wxyz, (1.0, 0.0, 0.0))
    cam_y = quat_rotate(rotation_wxyz, (0.0, 1.0, 0.0))
    cam_z = quat_rotate(rotation_wxyz, (0.0, 0.0, 1.0))
    local = (_dot(delta, cam_x), _dot(delta, cam_y), _dot(delta, cam_z))
    if local[2] >= -near_clip or local[2] <= -far_clip:
        return False
    half_width = abs(local[2]) * math.tan(0.5 * horizontal_fov_rad)
    return abs(local[0]) <= half_width and abs(local[1]) <= half_width


def _look_at_opengl_wxyz(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    world_up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[float, float, float, float]:
    cam_z = _normalize(_sub(eye, target))
    cam_x = _cross(world_up, cam_z)
    if _norm(cam_x) < 1.0e-6:
        cam_x = _cross((1.0, 0.0, 0.0), cam_z)
    cam_x = _normalize(cam_x)
    cam_y = _cross(cam_z, cam_x)
    return _axes_to_wxyz(cam_x, cam_y, cam_z)


def _axes_to_wxyz(
    x_axis: tuple[float, float, float],
    y_axis: tuple[float, float, float],
    z_axis: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    m00, m01, m02 = x_axis[0], y_axis[0], z_axis[0]
    m10, m11, m12 = x_axis[1], y_axis[1], z_axis[1]
    m20, m21, m22 = x_axis[2], y_axis[2], z_axis[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return (
            0.25 * scale,
            (m21 - m12) / scale,
            (m02 - m20) / scale,
            (m10 - m01) / scale,
        )
    if m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        return (
            (m21 - m12) / scale,
            0.25 * scale,
            (m01 + m10) / scale,
            (m02 + m20) / scale,
        )
    if m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        return (
            (m02 - m20) / scale,
            (m01 + m10) / scale,
            0.25 * scale,
            (m12 + m21) / scale,
        )
    scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
    return (
        (m10 - m01) / scale,
        (m02 + m20) / scale,
        (m12 + m21) / scale,
        0.25 * scale,
    )


def _pull_back_to_frame(
    eye: tuple[float, float, float],
    look_at: tuple[float, float, float],
    *,
    horizontal_fov_rad: float,
    near_clip: float,
    far_clip: float,
    workspace_half_xy: float,
    workspace_height: float,
) -> tuple[tuple[float, float, float], float]:
    offset = _sub(eye, look_at)
    radius = _norm(offset)
    if radius < 1.0e-6:
        return eye, 1.0
    direction = _normalize(offset)
    corners = _workspace_corners(
        look_at,
        workspace_half_xy=workspace_half_xy,
        workspace_height=workspace_height,
    )
    best_eye = eye
    best_scale = 1.0
    scale = 1.0
    while scale <= MAX_ORBIT_RADIUS_SCALE + 1.0e-9:
        candidate = _add(look_at, _scale_vec(direction, radius * scale))
        rotation = _look_at_opengl_wxyz(candidate, look_at)
        if all(
            _point_in_opengl_frustum(
                corner,
                candidate,
                rotation,
                horizontal_fov_rad=horizontal_fov_rad,
                near_clip=near_clip,
                far_clip=max(far_clip, FAR_CLIP_MARGIN * _norm(_sub(corner, candidate))),
            )
            for corner in corners
        ):
            return candidate, scale
        best_eye = candidate
        best_scale = scale
        scale += 0.05
    return best_eye, best_scale


def perturb_task(task, suite, look):
    """Copy task geometry; retain authored robot, materials, mass and timing."""
    if not look.scale:
        return task
    if '_eval_object_scale' in task:
        raise ValueError('Task geometry has already been perturbed')
    result = deepcopy(task)
    factor = look.effective_object_scale
    result['_eval_object_scale'] = factor
    if suite == 'tac_dynamic':
        episode = result['episode_config']
        for key in ('payload_scale', 'fixture_scale', 'gear_scale', 'base_scale', 'source_holder_scale'):
            if key in episode or key == 'source_holder_scale':
                episode[key] = float(episode.get(key, 1.0)) * factor
        # These are world-metre offsets, not asset-local grasp coordinates.
        scalar = ('success_xy_tolerance_m', 'success_z_tolerance_m', 'success_radial_tolerance_m',
                  'contact_offset_m', 'payload_contact_offset_m', 'fixture_contact_offset_m',
                  'rest_offset_m', 'fixture_mount_clearance_m', 'base_mount_clearance_m',
                  'target_fixture_opening_x_m', 'target_fixture_opening_y_m',
                  'target_fixture_wall_height_m')
        vectors = ('goal_offset_in_fixture_m', 'goal_offset_in_base_m', 'target_entry_local_m',
                   'bore_entry_offset_in_fixture_m', 'source_payload_offset_m',
                   'success_axial_range_m')
        for key in scalar:
            if key in episode:
                episode[key] *= factor
        for key in vectors:
            if key in episode:
                episode[key] = [v * factor for v in episode[key]]
    else:
        result['_eval_nominal_task'] = deepcopy(task)
        from benchmark.catalog import is_tabletop_task
        if is_tabletop_task(task):
            from benchmark.tasks.success import _tabletop_goal
            result['goals'][0] = deepcopy(_tabletop_goal(task))
        for spec in result['objects'].values():
            spec['scale'] = [v * factor for v in spec['scale']]
        # ref_local_offset and in-hand grasp points are already multiplied by
        # object.scale in TacMind. Scaling those here would apply the factor twice.
        for goal in result['goals']:
            for key in ('xy_threshold', 'height_threshold', 'height_diff'):
                if key in goal:
                    goal[key] *= factor
            for key in ('target_local_offset', 'insertion_entry_local', 'distance_range',
                        'height_range', 'vertical_separation_range', 'thread_engagement_range'):
                if key in goal:
                    goal[key] = [v * factor for v in goal[key]]
    return result


def perturb_reset_poses(task, poses, factor):
    """Scale local seating offsets after the nominal seeded sampler.

    World XY randomization and quaternions are preserved. Assemblies shrink
    about their own support roots, independent objects about the table plane.
    This also covers hard-coded precision reset geometry without duplicating it.
    """
    result = deepcopy(poses)
    surface = float(task['workspace_offset'][2])
    def scale_about(name, anchor):
        xyz = poses[name]['position']
        result[name]['position'] = [anchor[i] + (xyz[i] - anchor[i]) * factor for i in range(3)]
    for name, pose in poses.items():
        scale_about(name, (pose['position'][0], pose['position'][1], surface))
    for group in task.get('randomization_groups', {}).values():
        members = group['members']
        first = next(iter(members))
        anchor = [poses[first]['position'][i] - members[first][i] for i in range(3)]
        for name in members:
            scale_about(name, anchor)
    contract = task.get('reset_contract') or {}
    mode = contract.get('mode')
    anchor_name = None
    if mode in ('box_bottom', 'object_search'):
        anchor_name = contract['box']
        names = [contract['object']]
    elif mode == 'ball_sorting':
        anchor_name = contract['base']
        names = [contract[key] for key in ('golf', 'table_tennis', 'blank')]
    elif mode == 'blind_permutation':
        anchor_name = contract['station']
        names = [role['object'] for role in contract['roles'].values()]
    elif mode == 'exchange_layout':
        names = [name for group in contract['exchange_groups'] for name in group]
        # The station/deck/parts box is part of TacMind's scaled collision
        # assembly, so slot centres and token seating heights shrink together.
        anchor_name = next((name for name in ('blind_station', 'source_deck', 'parts_box')
                            if name in poses), None)
    if anchor_name is not None:
        anchor = result[anchor_name]['position']
        original_anchor = poses[anchor_name]['position']
        for name in names:
            result[name]['position'] = [anchor[i] + (poses[name]['position'][i] - original_anchor[i]) * factor
                                        for i in range(3)]
    return result
