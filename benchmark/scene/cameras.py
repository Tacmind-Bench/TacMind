"""Shared RGB camera poses for Tac tasks and the GelSight Mini example."""

CAMERAS = {
    "agentview": {
        "name": "standard cam",
        # TacBenchHub tabletop pose advanced 3.5 cm along its optical axis.
        "position": (1.0868705856925738, -5.400834584723402e-08, 0.5214930589756261),
        "rotation": (0.6182166934013367, 0.3432307541370392,
                     0.3432314395904541, 0.6182177066802979),
        "focal_length": 18.0,
        "horizontal_aperture": 15.0,
        "resolution": (448, 448),
        "clipping_range": (0.1, 1.9),
    },
    "wristview": {
        # Tabero 56a3b905: JointPositionLiberoCameraEnvCfg.eye_in_hand_cam,
        # relative to panda_hand, quaternion wxyz in the OpenGL convention.
        "position": (0.05, 0.0, 0.0),
        "rotation": (0.0, 0.707108, 0.707108, 0.0),
        "focal_length": 10.0,
        "horizontal_aperture": 20.955,
        "resolution": (480, 480),
        "clipping_range": (0.02, 3.0),
    },
}


def camera_specs(suite: str | None = None) -> dict:
    """Keep Dynamic's translated agent view separate from the shared cameras."""
    if suite in ("dynamic", "tac_dynamic"):
        return {**CAMERAS, "agentview": {
                "name": "dynamic standard cam", "position": (1.412657, 0.30, 0.635665),
                "rotation": (0.58274522, 0.40050969, 0.40050969, 0.58274522),
                "focal_length": 25.0, "horizontal_aperture": 20.955,
                "resolution": CAMERAS["wristview"]["resolution"],
                "clipping_range": (0.05, 10.0)}}
    return CAMERAS


def camera_frame_report(points, camera=None, *, base_position=(0, 0, 0),
                        base_rotation=(1, 0, 0, 0), border=0.02):
    """Project world points with the robot-relative OpenGL camera calibration."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    camera = CAMERAS["agentview"] if camera is None else camera
    if not 0 <= border < .5:
        raise ValueError("Camera border must be in [0, 0.5)")
    def rotation(q):
        return Rotation.from_quat((*q[1:], q[0]))
    base = rotation(base_rotation)
    origin = base.apply(camera["position"])+base_position
    local = (base*rotation(camera["rotation"])).inv().apply(np.asarray(points)-origin)
    depth = -local[:, 2]
    slope = camera["horizontal_aperture"]/camera["focal_length"]
    aspect = camera["resolution"][0]/camera["resolution"][1]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = .5+local[:, :2]/depth[:, None]/[slope, slope/aspect]
    near, far = camera["clipping_range"]
    valid = (np.isfinite(local).all(axis=1) & (depth>=near) & (depth<=far)
             & (uv>=border).all(axis=1) & (uv<=1-border).all(axis=1))
    return {"passed": bool(len(valid) and valid.all()), "border_fraction": border,
            "uv": uv.tolist(), "depth_m": depth.tolist(), "inside": valid.tolist(),
            "camera_world_position_m": origin.tolist()}


def camera_metadata(scene) -> dict:
    """Read actual configured camera optics and offsets for recording/evaluation."""
    metadata = {}
    for name in CAMERAS:
        camera = getattr(scene, name)
        metadata[name] = {
            "position": list(camera.offset.pos),
            "rotation": list(camera.offset.rot),
            "convention": camera.offset.convention,
            "focal_length": camera.spawn.focal_length,
            "horizontal_aperture": camera.spawn.horizontal_aperture,
            "resolution": [camera.width, camera.height],
            "clipping_range": list(camera.spawn.clipping_range),
        }
    return metadata


def configure_camera_shapes(scene, shapes):
    """Restore recorded RGB sizes without changing calibrated tactile resolution."""
    for name, shape in shapes.items():
        sensor = getattr(scene, name)
        if not name.startswith("gsmini_"):
            sensor.height, sensor.width = shape[:2]


def camera_cfg(name: str, prim_path: str, *, suite: str | None = None,
               override: dict | None = None):
    """Use Isaac Lab's tiled RGB camera for simulation observations."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import TiledCameraCfg

    spec = camera_specs(suite)[name]
    if override:
        spec = {**spec, **override}
        # JSON round-trips tuples as lists, and Isaac Lab needs typed sequences.
        spec["position"] = tuple(float(v) for v in spec["position"])
        spec["rotation"] = tuple(float(v) for v in spec["rotation"])
        spec["clipping_range"] = tuple(float(v) for v in spec["clipping_range"])
        spec["resolution"] = tuple(int(v) for v in spec["resolution"])
    return TiledCameraCfg(
        prim_path=prim_path,
        width=spec["resolution"][0], height=spec["resolution"][1],
        data_types=["rgb"],
        update_latest_camera_pose=name == "wristview",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=spec["focal_length"],
            horizontal_aperture=spec["horizontal_aperture"],
            clipping_range=spec["clipping_range"],
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=spec["position"], rot=spec["rotation"], convention="opengl",
        ),
    )


def configure_rgb_cameras(scene, robot_path="{ENV_REGEX_NS}/Robot", *,
                          suite: str | None = None,
                          agentview_override: dict | None = None) -> None:
    """Attach the shared RGB pair to a cloned or authored robot.

    ``agentview_override`` lets a suite whose parts are much smaller than the
    tabletop-scale workspace place its own agent view without forking the shared
    camera; the wrist view and every other camera stay identical.
    """
    scene.agentview = camera_cfg("agentview", f"{robot_path}/agentview", suite=suite,
                                 override=agentview_override)
    scene.wristview = camera_cfg("wristview", f"{robot_path}/panda_hand/wristview", suite=suite)


def _raw_rgb(env, sensor_cfg):
    """Read once; ObservationManager copies the sample into its observation buffer."""
    import torch

    sensor = env.scene.sensors[sensor_cfg.name]
    output = sensor.data.output
    if "rgb" in output:
        return output["rgb"]
    # ObservationManager sizes its terms before the first render.
    return torch.zeros((env.num_envs, sensor.cfg.height, sensor.cfg.width, 3),
                       device=env.device, dtype=torch.uint8)


def vision_observations_cfg():
    """Keep raw RGB observation terms consistent across tactile backends."""
    from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
    from isaaclab.utils import configclass

    @configclass
    class VisionObservationsCfg(ObservationGroupCfg):
        agentview = ObservationTermCfg(
            func=_raw_rgb, params={"sensor_cfg": SceneEntityCfg("agentview")},
        )
        wristview = ObservationTermCfg(
            func=_raw_rgb, params={"sensor_cfg": SceneEntityCfg("wristview")},
        )
        concatenate_terms = False
        enable_corruption = False

    return VisionObservationsCfg()


def add_preview_cameras(stage, robot_path: str, *, suite: str | None = None) -> None:
    """Author the same camera optics in a preview's session layer."""
    from pxr import Gf, Usd, UsdGeom

    specs = camera_specs(suite)
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for name, path in (("agentview", f"{robot_path}/agentview"),
                           ("wristview", f"{robot_path}/panda_hand/wristview")):
            spec = specs[name]
            camera = UsdGeom.Camera.Define(stage, path)
            camera.GetPrim().SetDisplayName(spec.get("name", name))
            ops = camera.GetOrderedXformOps()
            if ops:
                if [op.GetOpName() for op in ops] != ["xformOp:translate", "xformOp:orient"]:
                    raise ValueError(f"Unexpected camera transform at {path}")
                translate, orient = ops
            else:
                translate, orient = camera.AddTranslateOp(), camera.AddOrientOp()
            translate.Set(Gf.Vec3d(*spec["position"]))
            w, x, y, z = spec["rotation"]
            orient.Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
            camera.CreateFocalLengthAttr(spec["focal_length"])
            camera.CreateHorizontalApertureAttr(spec["horizontal_aperture"])
            camera.CreateVerticalApertureAttr(
                spec["horizontal_aperture"] * spec["resolution"][1] / spec["resolution"][0]
            )
            camera.CreateClippingRangeAttr(Gf.Vec2f(*spec["clipping_range"]))
