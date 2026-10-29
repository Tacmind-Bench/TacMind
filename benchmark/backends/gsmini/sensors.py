"""Shared lift task with the physical GelSight Mini gripper and Taxim RGB.

Import after AppLauncher(enable_cameras=True). The environment exposes tactile
RGB.
"""


from pathlib import Path
from copy import deepcopy
from functools import lru_cache

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.sensors import ContactSensor
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from pxr import Usd, UsdGeom

from benchmark.backends.gsmini.robot import FrankaLiftEnvCfg, _lift_test_object_spawn_kwargs, _pad_centers_world

from benchmark.catalog import ROOT, ASSETS, ROBOT_ASSET
from benchmark.scene.cameras import configure_rgb_cameras, vision_observations_cfg
from benchmark.backends.gsmini.tacex import GelSightSensor, GelSightSensorCfg
from benchmark.backends.gsmini.tacex.simulation_approaches.gpu_taxim import TaximSimulatorCfg


def planner_robot(env, robot_usd_path=None, attached_sphere_count=4):
    """Return an independent planner config with cached, conservative sensor coverage."""
    path = str(Path(robot_usd_path or env.cfg.scene.robot.spawn.usd_path).resolve())
    stage = Usd.Stage.Open(str(path))
    # Referenced meshes and payloads can change without changing the root USD.
    signature = tuple(sorted((layer.realPath, Path(layer.realPath).stat().st_mtime_ns,
                              Path(layer.realPath).stat().st_size)
                             for layer in stage.GetUsedLayers() if layer.realPath))
    config = deepcopy(_planner_robot_geometry(path, signature))
    extra = config["robot_cfg"]["kinematics"]["extra_collision_spheres"]
    extra["attached_object"] = max(extra["attached_object"], int(attached_sphere_count))
    return config


@lru_cache(maxsize=4)
def _planner_robot_geometry(robot_usd_path, signature):
    """Conservatively cover the added colliders in the existing finger frames.

    Only the planner is modified. The physical USD meshes/materials remain intact.
    Use the source stage at its authored zero pose so render/PhysX synchronization
    cannot change the relationship between a finger and its fixed sensor assembly.
    """
    import itertools
    import math
    import numpy as np
    from pxr import UsdPhysics
    from curobo.config_io import load_yaml, join_path
    from curobo.content import get_robot_configs_path

    config = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    stage = Usd.Stage.Open(str(robot_usd_path))
    transforms = UsdGeom.XformCache()
    bounds = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=False,
    )
    for side in ("left", "right"):
        finger = f"panda_{side}finger"
        inverse = transforms.GetLocalToWorldTransform(
            stage.GetPrimAtPath(f"/panda/{finger}")
        ).GetInverse()
        spheres = config["robot_cfg"]["kinematics"]["collision_spheres"][finger]
        count = 0
        for body in (f"gelsight_mini_case_{side}", f"gelpad_{side}"):
            for prim in Usd.PrimRange(stage.GetPrimAtPath(f"/panda/{body}")):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                    continue
                box = bounds.ComputeLocalBound(prim)
                box.Transform(transforms.GetLocalToWorldTransform(prim) * inverse)
                extent = box.ComputeAlignedRange()
                low, high = np.array(extent.GetMin()), np.array(extent.GetMax())
                cells = np.maximum(1, np.ceil((high - low) / 0.006).astype(int))
                size = (high - low) / cells
                radius = float(np.linalg.norm(size) / 2)
                if not math.isfinite(radius) or radius <= 0:
                    raise RuntimeError(f"Invalid collision bounds: {prim.GetPath()}")
                for index in itertools.product(*(range(int(n)) for n in cells)):
                    center = low + (np.array(index) + 0.5) * size
                    spheres.append({"center": center.tolist(), "radius": radius})
                    count += 1
        if not count:
            raise RuntimeError(f"Missing {side} sensor collision geometry")
        print(f"[COLLISION] {side}: {count} sensor spheres", flush=True)
    return config


class PhysicsStepContactSensor(ContactSensor):
    """Reuse native force/history sampling without querying uniform CUDA flags."""

    _all_outdated = None

    def _initialize_impl(self):
        self._all_outdated = None
        super()._initialize_impl()
        if self.cfg.update_period > 0.0:
            # Keep periodic samples due at control boundaries after float32
            # accumulation loses more than SensorBase's scheduling tolerance.
            self._timestamp = self._timestamp.double()
            self._timestamp_last_update = self._timestamp_last_update.double()

    def reset(self, env_ids=None):
        # Partial resets retain Isaac Lab's per-environment scheduling.
        self._all_outdated = None
        super().reset(env_ids)

    def update(self, dt, force_recompute=False):
        self._all_outdated = True if self.cfg.update_period == 0.0 and dt >= 0.0 else None
        if self._all_outdated and (force_recompute or self._is_visualizing or self.cfg.history_length > 0):
            # Every environment is due: retain native timestamps and history,
            # without computing a CUDA scheduling mask that is immediately cleared.
            self._timestamp += dt
            self._update_outdated_buffers()
        else:
            super().update(dt, force_recompute=force_recompute)

    def _update_outdated_buffers(self):
        if self.cfg.update_period != 0.0 or self._all_outdated is None:
            super()._update_outdated_buffers()
        elif self._all_outdated:
            self._update_buffers_impl(range(self._num_envs))
            self._timestamp_last_update.copy_(self._timestamp)
            self._is_outdated.fill_(False)
        self._all_outdated = False


class ControlStepGelSightSensor(GelSightSensor):
    """Keep parent tactile frames on their original control schedule."""

    def _initialize_impl(self):
        super()._initialize_impl()
        if self.cfg.update_period > 0.0:
            self._timestamp = self._timestamp.double()
            self._timestamp_last_update = self._timestamp_last_update.double()


def configure_contact_sensors(cfg, filters, *, robot_path="{ENV_REGEX_NS}/Robot",
                              marker_target_path=None):
    """Observe gel and housing separately, retaining every physical-step sample."""
    from isaaclab.sensors import ContactSensorCfg

    for side in ("left", "right"):
        for name, body in ((f"gsmini_{side}_contact", f"gelpad_{side}"),
                           (f"gsmini_{side}_case_contact", f"gelsight_mini_case_{side}")):
            setattr(cfg.scene, name, ContactSensorCfg(
                class_type=PhysicsStepContactSensor,
                prim_path=f"{robot_path}/{body}", update_period=0.0,
                history_length=cfg.decimation, filter_prim_paths_expr=list(filters),
            ))
        # Read native friction at the control boundary; retain existing normal
        # sensors and their physics-rate histories without extra buffer queries.
        setattr(cfg.scene, f"gsmini_{side}_surface_contact", ContactSensorCfg(
            class_type=PhysicsStepContactSensor,
            prim_path=f"{robot_path}/gelpad_{side}",
            update_period=cfg.sim.dt*cfg.decimation, history_length=0,
            filter_prim_paths_expr=list(filters), track_friction_forces=True,
            max_contact_data_count_per_prim=64))
    if getattr(cfg.scene, 'gsmini_left', None) is not None:
        if marker_target_path is None:
            target = getattr(cfg.scene, 'object', None)
            marker_target_path = target.prim_path if target is not None else filters[0]
        configure_marker_motion(cfg, target_path=marker_target_path)


def configure_marker_motion(cfg, *, target_path=None, target_offset=(0.0, 0.0, 0.0), pose_motion=False):
    """Configure the Taxim-coupled FOTS marker stream without changing raw RGB."""
    from isaaclab.sensors import FrameTransformerCfg
    from benchmark.backends.gsmini.tacex.simulation_approaches.fots import FOTSMarkerSimulatorCfg
    from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

    env_namespace = "/World/envs/env_.*"
    for side in ("left", "right"):
        sensor = getattr(cfg.scene, f"gsmini_{side}")
        if "marker_motion" not in sensor.data_types:
            sensor.data_types = [*sensor.data_types, "marker_motion"]
        sensor.marker_motion_sim_cfg = FOTSMarkerSimulatorCfg(
            frame_transformer_cfg=FrameTransformerCfg(
                prim_path=sensor.prim_path.replace("{ENV_REGEX_NS}", env_namespace),
                target_frames=[FrameTransformerCfg.FrameCfg(
                    prim_path=(target_path or cfg.scene.object.prim_path).replace(
                        "{ENV_REGEX_NS}", env_namespace,
                    ),
                    offset=OffsetCfg(pos=target_offset),
                )],
            ),
        )
        sensor.marker_motion_sim_cfg.frame_transformer_cfg.visualizer_cfg.markers["frame"].usd_path = str(
            ASSETS / "objects/frame_prim/frame_prim.usd")
        if pose_motion:
            import numpy as np
            from scipy.spatial.transform import Rotation

            stage = Usd.Stage.Open(str(cfg.scene.robot.spawn.usd_path))
            camera = UsdGeom.Camera(stage.GetPrimAtPath(f"/panda/gelsight_mini_case_{side}/Camera"))
            mount = UsdGeom.Xformable(camera).GetLocalTransformation()
            # OpenGL camera axes -> image x-right/y-down/z-forward.
            rotation = Rotation.from_matrix(np.asarray(mount)[:3, :3].T @ np.diag([1, -1, -1]))
            x, y, z, w = rotation.as_quat()
            marker = sensor.marker_motion_sim_cfg
            marker.motion_source = "target_pose"
            marker.frame_transformer_cfg.source_frame_offset = OffsetCfg(
                pos=tuple(mount.ExtractTranslation()), rot=(w, x, y, z))
            # Use the authored camera's projection at the undeformed gel surface.
            # The legacy FOTS coefficient is not this camera's pixels per mm.
            gel_distance_mm = 1000 * (sensor.optical_sim_cfg.gelpad_to_camera_min_distance
                                      + sensor.optical_sim_cfg.gelpad_height)
            marker.mm_to_pixel = sensor.sensor_camera_cfg.resolution[0] * (
                camera.GetFocalLengthAttr().Get() / camera.GetHorizontalApertureAttr().Get()) / gel_distance_mm


def tactile_rgb(env, sensor_name):
    return env.scene[sensor_name].data.output["tactile_rgb"]


def tactile_marker_motion(env, sensor_name):
    return env.scene[sensor_name].data.output["marker_motion"]


def pad_centers_world(env):
    return _pad_centers_world(env, ("gelpad_left", "gelpad_right"),
                              env.cfg.gsmini_pad_centers_local)


@configclass
class TactileObservationsCfg(ObservationGroupCfg):
    left = ObservationTermCfg(func=tactile_rgb, params={"sensor_name": "gsmini_left"})
    right = ObservationTermCfg(func=tactile_rgb, params={"sensor_name": "gsmini_right"})
    concatenate_terms = False
    enable_corruption = False


@configclass
class FrankaGsminiLiftEnvCfg(FrankaLiftEnvCfg):
    gsmini_pad_centers_local: tuple = ()

    def __post_init__(self):
        super().__post_init__()
        # Shared 20 Hz control/observations; retain the 100 Hz physics steps.
        self.decimation = 5
        self.sim.render_interval = self.decimation
        self.events.reset_object_position.params["robot_config"] = planner_robot
        # A narrow curved surface leaves visible contact boundaries on the gel.
        # Keep enough height for the sensor housing to clear the table.
        self.scene.object.spawn = sim_utils.CylinderCfg(
            radius=0.009, height=0.040,
            **_lift_test_object_spawn_kwargs(self.scene.object.spawn.rigid_props),
        )
        self.scene.object.init_state.pos = (0.5, 0.0, 0.022)
        self.scene.robot.spawn.usd_path = str(ASSETS / ROBOT_ASSET)
        self.scene.robot.spawn.activate_contact_sensors = True
        configure_contact_sensors(self, [self.scene.object.prim_path])
        configure_rgb_cameras(self.scene)
        # Center the grasp on the two gel surfaces, not the old Panda fingertips.
        stage = Usd.Stage.Open(str(self.scene.robot.spawn.usd_path))
        transforms = UsdGeom.XformCache()
        hand_inverse = transforms.GetLocalToWorldTransform(
            stage.GetPrimAtPath("/panda/panda_hand")
        ).GetInverse()
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        centers = [hand_inverse.Transform(bounds.ComputeWorldBound(
            stage.GetPrimAtPath(f"/panda/gelpad_{side}/mesh")
        ).ComputeCentroid()) for side in ("left", "right")]
        self.gsmini_pad_centers_local = tuple(tuple(
            transforms.GetLocalToWorldTransform(stage.GetPrimAtPath(
                f"/panda/gelpad_{side}")).GetInverse().Transform(
                    bounds.ComputeWorldBound(stage.GetPrimAtPath(
                        f"/panda/gelpad_{side}/mesh")).ComputeCentroid()))
            for side in ("left", "right"))
        self.scene.ee_frame.target_frames[0].offset.pos = tuple((centers[0] + centers[1]) / 2)
        # Gel centers are separated even when the Panda finger joints read zero.
        self.grasp_width_offset = abs(centers[0][1] - centers[1][1])
        for side in ("left", "right"):
            sensor = GelSightSensorCfg(
                class_type=ControlStepGelSightSensor,
                prim_path=f"{{ENV_REGEX_NS}}/Robot/gelsight_mini_case_{side}",
                update_period=self.sim.dt * self.decimation,
                data_types=["tactile_rgb"],
                sensor_camera_cfg=GelSightSensorCfg.SensorCameraCfg(
                    resolution=(320, 240), data_types=["depth"], clipping_range=(0.024, 0.029),
                    update_latest_camera_pose=False,
                ),
                optical_sim_cfg=TaximSimulatorCfg(
                    calib_folder_path=str(ROOT / "assets/sensors/GelSight_Mini/calibs/640x480"),
                    gelpad_height=0.0045, gelpad_to_camera_min_distance=0.024,
                    with_shadow=False, tactile_img_res=(320, 240),
                ),
                marker_motion_sim_cfg=None,
                debug_vis=False,
            )
            setattr(self.scene, f"gsmini_{side}", sensor)
        configure_marker_motion(self)
        self.observations.tactile = TactileObservationsCfg()
        self.observations.vision = vision_observations_cfg()
