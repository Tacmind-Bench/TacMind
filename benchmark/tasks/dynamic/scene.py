"""TacDynamic scene construction and physics configuration."""

from __future__ import annotations


import math
import random
from copy import deepcopy

from benchmark.expert.capabilities import supported_task_ids

PLATFORM_PATH = "/World/BeltTaskPlatform"
PLATFORM_FIXTURE_JOINT_PATH = "/World/TacDynamicPlatformFixtureJoint"
PLATFORM_SOURCE_HOLDER_JOINT_PATH = "/World/TacDynamicPlatformSourceHolderJoint"
# The existing platform is a solid metallic cuboid, modeled at aluminum density.
PLATFORM_DENSITY_KG_M3 = 2700.0
GSMINI_PAD_FRICTION = 0.5
TASK_IDS = supported_task_ids("dynamic", "gsmini")


def resolve_gsmini_task(task: dict) -> dict:
    """Apply the task's authored GelSight-specific runtime contract."""
    resolved = deepcopy(task)
    episode = resolved.get("episode_config") or {}
    override = (resolved.get("sensor_overrides") or {}).get("gsmini") or {}

    def merge(target, values):
        for key, value in values.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = deepcopy(value)

    merge(episode, override)
    resolved["episode_config"] = episode
    resolved["sensor_profile"] = "gsmini"
    return resolved


def configure_gsmini_pad_material(stage, robot_path: str) -> None:
    """Give the physical gel pads a dissipative rubber contact in this scene only."""
    from pxr import Usd

    material = stage.GetPrimAtPath(f"{robot_path}/PhysicsMaterial")
    if not material:
        raise ValueError(f"Missing GSmini pad material: {robot_path}/PhysicsMaterial")
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for name, value in (("physics:staticFriction", GSMINI_PAD_FRICTION),
                            ("physics:dynamicFriction", GSMINI_PAD_FRICTION),
                            ("physics:restitution", 0.0)):
            attribute = material.GetAttribute(name)
            if not attribute:
                raise ValueError(f"Missing GSmini pad material property: {name}")
            attribute.Set(value)
    print(f"[PHYSICS] GSmini pad friction={GSMINI_PAD_FRICTION:.1f}, restitution=0", flush=True)


def dynamic_goal_quaternion(task: dict) -> tuple[float, float, float, float]:
    """Use the authored goal orientation or align object-local Z to the bore axis."""
    episode = task["episode_config"]
    authored = episode.get("expected_object_local_quat_wxyz")
    if authored is not None:
        if len(authored) != 4 or not all(math.isfinite(float(value)) for value in authored):
            raise ValueError("Invalid authored dynamic goal quaternion")
        return tuple(map(float, authored))

    axis = tuple(map(float, episode["bore_axis_local"]))
    if len(axis) != 3 or not all(math.isfinite(value) for value in axis):
        raise ValueError("Invalid dynamic bore axis")
    length = math.sqrt(sum(value * value for value in axis))
    if length <= 1.0e-8:
        raise ValueError("Dynamic bore axis must be nonzero")
    ax, ay, az = (value / length for value in axis)
    if az <= -1.0 + 1.0e-8:
        return 0.0, 1.0, 0.0, 0.0
    scale = math.sqrt(2.0 * (1.0 + az))
    return scale / 2.0, -ay / scale, ax / scale, 0.0


def dynamic_scene_roles(task: dict) -> dict[str, str | None]:
    """Map task assets to moving and tabletop roles from the task profile."""
    episode = task["episode_config"]
    in_hand = episode.get("in_hand_reset", {}).get("mode") == "in_hand"
    if episode["profile"] == "table_to_platform":
        return {
            "moving": "fixture",
            "tabletop": None if in_hand else "object",
            "static_fixture": None,
            "source_holder": None if in_hand else episode.get("source_holder_type"),
        }
    return {
        "moving": "object",
        "tabletop": None,
        "static_fixture": "fixture",
        "source_holder": episode.get("source_holder_type"),
    }


def dynamic_asset_scale(task: dict, role: str) -> tuple[float, float, float]:
    """Return the authored family scale, including its task-specific geometry rule."""
    episode = task["episode_config"]
    if role not in {"object", "fixture"}:
        raise ValueError(f"Unknown TacDynamic asset role: {role}")
    scale = float(episode[f"{role}_scale"] if role == "fixture"
                  else episode["payload_scale"])
    if episode["object_family"] == "key":
        scale *= 1.5
    return (scale,) * 3


def dynamic_payload_task(task: dict) -> dict:
    """Share the authored payload and in-hand reset contract across collection and evaluation."""
    episode = task["episode_config"]
    in_hand = episode.get("in_hand_reset", {}).get("mode") == "in_hand"
    reset = deepcopy(episode.get("in_hand_reset", {})) if in_hand else {}
    if in_hand:
        reset["grasp_point_local"] = episode["in_hand_grasp_local_m"]
        reset["in_hand_object_local_quat_wxyz"] = episode["in_hand_object_local_quat_wxyz"]
    return {
        "obj_of_interest": ["object"],
        "objects": {"object": {
            "type": episode["payload_asset_type"],
            "scale": dynamic_asset_scale(task, "object"),
        }},
        "goals": [{
            "target": "fixture",
            "target_local_offset": episode["goal_offset_in_fixture_m"],
            "expected_object_local_quat": dynamic_goal_quaternion(task),
            "xy_threshold": episode.get("success_xy_tolerance_m", 0.004),
        }],
        "reset_contract": reset,
    }


def dynamic_source_payload_position(task: dict, holder_position) -> tuple[float, ...]:
    """Place a source payload at its authored offset in the rotated holder frame."""
    from pxr import Gf

    episode = task["episode_config"]
    yaw = float(episode.get("base_yaw_deg", 0.0))
    offset = Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw).TransformDir(
        Gf.Vec3d(*episode["source_payload_offset_m"]),
    )
    return tuple(float(holder_position[i]) + float(offset[i]) for i in range(3))


def _belt_range(stage):
    from pxr import Usd, UsdGeom

    visual = stage.GetPrimAtPath("/World/ConveyorBeltSurface")
    if not visual:
        raise ValueError("Missing /World/ConveyorBeltSurface")
    bounds = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render,
                                 UsdGeom.Tokens.proxy], useExtentsHint=True,
    ).ComputeWorldBound(visual).ComputeAlignedRange()
    low, high = bounds.GetMin(), bounds.GetMax()
    if any(not math.isfinite(float(high[i] - low[i])) or high[i] <= low[i]
           for i in range(3)):
        raise ValueError("Invalid conveyor surface bounds")
    return low, high


def platform_reset_pose(stage, task: dict, reset: dict,
                        drop_clearance_m: float = 0.0) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return the platform center and dimensions from the seeded task reset."""
    episode = task["episode_config"]
    low, high = _belt_range(stage)
    size = tuple(map(float, episode["platform_size_m"]))
    if len(size) != 3 or not all(math.isfinite(value) and value > 0 for value in size):
        raise ValueError(f"Invalid task platform size: {size}")
    if not math.isfinite(drop_clearance_m) or drop_clearance_m < 0:
        raise ValueError(f"Invalid platform drop clearance: {drop_clearance_m}")
    center = (
        float((low[0] + high[0]) / 2 + reset["platform_x_offset_m"]),
        float(high[1] - size[1] / 2 - 0.02 - reset["platform_entry_phase_m"]),
        float(high[2] + size[2] / 2 + drop_clearance_m),
    )
    return center, size


def configure_conveyor_platform(sim, task: dict, reset: dict,
                                drop_clearance_m: float = 0.01) -> tuple[
                                    tuple[float, ...], tuple[float, ...]]:
    """Reset the dynamic platform above the belt so gravity settles it onto the belt."""
    import isaaclab.sim as sim_utils
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = sim.stage if hasattr(sim, "stage") else sim
    platform = stage.GetPrimAtPath(PLATFORM_PATH)
    rigid_body = UsdPhysics.RigidBodyAPI(platform)
    collision = UsdPhysics.CollisionAPI(platform)
    if not platform or not rigid_body or not collision:
        raise ValueError(f"Missing physical platform at {PLATFORM_PATH}")
    center, size = platform_reset_pose(stage, task, reset, drop_clearance_m)

    with Usd.EditContext(stage, stage.GetSessionLayer()):
        # Collection reset positions require a stationary tabletop support.
        sim_utils.modify_rigid_body_properties(
            "/World/BigTable", sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            stage=stage,
        )
        for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/BigTable")):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        # Sample walls retain 100-unit extents despite metre-scale cube sizes.
        # Refresh bounds without changing the physical guard dimensions.
        for belt in ("conveyor", "conveyor_01"):
            for side in ("left_wall", "right_wall"):
                wall = UsdGeom.Cube.Get(stage, f"/World/Ur10Table/{belt}/{side}")
                if not wall:
                    raise ValueError(f"Missing conveyor side guard: {belt}/{side}")
                wall.CreateExtentAttr(UsdGeom.Boundable.ComputeExtentFromPlugins(
                    wall, Usd.TimeCode.Default()))

        translate = platform.GetAttribute("xformOp:translate")
        scale = platform.GetAttribute("xformOp:scale")
        if not translate or not scale:
            raise ValueError(f"Platform requires translate and scale ops: {PLATFORM_PATH}")
        translate.Set(Gf.Vec3d(*center))
        scale.Set(Gf.Vec3d(*size))
        rigid_body.GetKinematicEnabledAttr().Set(False)
        collision.GetCollisionEnabledAttr().Set(True)
        mass_cfg = sim_utils.MassPropertiesCfg(density=PLATFORM_DENSITY_KG_M3)
        sim_utils.define_mass_properties(str(platform.GetPath()), mass_cfg, stage=stage)
    return center, size


def configure_moving_asset(stage, prim_path: str, mass_kg: float | None = None) -> str:
    """Make a referenced platform load respond to contact without editing its source USD."""
    import isaaclab.sim as sim_utils
    from pxr import Usd, UsdPhysics

    root = stage.GetPrimAtPath(prim_path)
    if not root:
        raise ValueError(f"Missing moving platform asset: {prim_path}")
    if mass_kg is not None and (not math.isfinite(mass_kg) or mass_kg <= 0):
        raise ValueError(f"Invalid moving asset mass: {mass_kg}")

    with Usd.EditContext(stage, stage.GetSessionLayer()):
        root.SetInstanceable(False)
        bodies = [prim for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                  if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        colliders = [prim for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                     if prim.HasAPI(UsdPhysics.CollisionAPI)]
        if len(bodies) != 1 or not colliders:
            raise ValueError(f"Moving asset must have one rigid body and colliders: {prim_path}")
        body_path = str(bodies[0].GetPath())
        rigid_cfg = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False, disable_gravity=False,
        )
        sim_utils.modify_rigid_body_properties(body_path, rigid_cfg, stage=stage)
        body = stage.GetPrimAtPath(body_path)
        if UsdPhysics.RigidBodyAPI(body).GetKinematicEnabledAttr().Get():
            raise ValueError(f"Could not make moving asset dynamic: {body_path}")
        if mass_kg is not None:
            mass_cfg = sim_utils.MassPropertiesCfg(mass=float(mass_kg))
            sim_utils.define_mass_properties(body_path, mass_cfg, stage=stage)
            mass = UsdPhysics.MassAPI(body).GetMassAttr().Get()
            if not math.isclose(mass, mass_kg, rel_tol=1e-6, abs_tol=1e-8):
                raise ValueError(f"Could not set moving asset mass: {body_path}")
    return body_path


def configure_source_holder_overrides(stage, prim_path: str, task: dict) -> None:
    """Apply authored source-holder friction and geometry overrides in the session layer."""
    import isaaclab.sim as sim_utils
    from pxr import Usd, UsdPhysics

    episode = task["episode_config"]
    friction = (episode.get("source_holder_static_friction"),
                episode.get("source_holder_dynamic_friction"))
    opening = (episode.get("source_holder_opening_x_m"),
               episode.get("source_holder_opening_y_m"))
    wall_height = episode.get("source_holder_wall_height_m")
    if all(value is None for value in (*friction, *opening, wall_height)):
        return

    root = stage.GetPrimAtPath(prim_path)
    if not root:
        raise ValueError(f"Missing source holder: {prim_path}")
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        root.SetInstanceable(False)
        if any(value is not None for value in (*opening, wall_height)):
            if episode.get("source_holder_type") != "hdmi_slot":
                raise ValueError("HDMI source-holder geometry overrides require hdmi_slot")
            holder_scale = float(episode.get("source_holder_scale", 1.0))
            if not math.isfinite(holder_scale) or holder_scale <= 0:
                raise ValueError(f"Invalid source-holder scale: {holder_scale}")
            import numpy as np
            from pxr import UsdGeom, Vt
            from benchmark.catalog import object_asset

            source = Usd.Stage.Open(str(object_asset("hdmi_slot")))
            source_root = source.GetDefaultPrim().GetPath()
            source_visual = UsdGeom.Mesh(source.GetPrimAtPath(str(source_root) + "/mesh"))
            vertices = np.asarray(source_visual.GetPointsAttr().Get())
            outer = np.max(np.abs(vertices[:, :2]), axis=0)
            floor_top = 0.00261779735
            rim_top = float(vertices[:, 2].max())
            requested_openings = [None if value is None else float(value) / holder_scale
                                  for value in opening]
            for axis, requested in enumerate(requested_openings):
                if requested is not None and (not math.isfinite(requested)
                                             or not 0 < requested < 2 * outer[axis]):
                    raise ValueError(f"Invalid HDMI source-holder opening: {requested}")
            height = None if wall_height is None else float(wall_height) / holder_scale
            if height is not None and (not math.isfinite(height) or height <= 0):
                raise ValueError(f"Invalid HDMI source-holder wall height: {height}")
            # Read the entry layer each time, so repeated setup is idempotent.
            # Keep the authored dimensions and deform visual/collision surfaces together.
            for original in source.Traverse():
                if not original.IsA(UsdGeom.Mesh):
                    continue
                enabled = (original.HasAPI(UsdPhysics.CollisionAPI)
                           and UsdPhysics.CollisionAPI(original).GetCollisionEnabledAttr().Get() is not False)
                if not enabled and UsdGeom.Imageable(original).ComputeVisibility() == "invisible":
                    continue
                points = np.asarray(UsdGeom.Mesh(original).GetPointsAttr().Get()).copy()
                for axis, inner in enumerate((0.003, 0.008)):
                    requested = requested_openings[axis]
                    if requested is None:
                        continue
                    distance = np.abs(points[:, axis])
                    half = requested / 2
                    stretched = np.where(distance <= inner, distance * half / inner,
                                         half + (distance - inner) * (outer[axis] - half)
                                         / (outer[axis] - inner))
                    points[:, axis] = np.sign(points[:, axis]) * stretched
                if height is not None:
                    above = points[:, 2] > floor_top
                    points[above, 2] = floor_top + (points[above, 2] - floor_top) * height / (rim_top - floor_top)
                relative = original.GetPath().MakeRelativePath(source_root)
                mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f"{prim_path}/Asset/{relative}"))
                if not mesh:
                    raise ValueError(f"Missing HDMI source-holder mesh: {relative}")
                mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
                mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(
                    np.asarray([points.min(0), points.max(0)], dtype=np.float32)))
                if not enabled:
                    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
                    normals, start = [], 0
                    for count in mesh.GetFaceVertexCountsAttr().Get():
                        polygon = points[indices[start:start + count]].astype(np.float64)
                        normal = np.cross(polygon, np.roll(polygon, -1, axis=0)).sum(0)
                        length = np.linalg.norm(normal)
                        if length > 0:
                            normal /= length
                        normals.extend([normal] * count)
                        start += count
                    mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(normals, dtype=np.float32)))
                    mesh.SetNormalsInterpolation("faceVarying")

        if any(value is not None for value in friction):
            if any(value is None for value in friction):
                raise ValueError("Source-holder friction requires static and dynamic values")
            static_friction, dynamic_friction = map(float, friction)
            if (not math.isfinite(static_friction) or static_friction < 0
                    or not math.isfinite(dynamic_friction) or dynamic_friction < 0):
                raise ValueError("Invalid source-holder friction")
            colliders = [prim for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                         if prim.HasAPI(UsdPhysics.CollisionAPI)]
            if not colliders:
                raise ValueError(f"Source holder has no colliders: {prim_path}")
            material_path = f"{prim_path}/TacMindPhysicsMaterial"
            material_cfg = sim_utils.RigidBodyMaterialCfg(
                static_friction=static_friction,
                dynamic_friction=dynamic_friction,
                restitution=0.0,
                friction_combine_mode="min",
                restitution_combine_mode="min",
            )
            with sim_utils.use_stage(stage):
                material_cfg.func(material_path, material_cfg)
                for collider in colliders:
                    sim_utils.bind_physics_material(
                        str(collider.GetPath()), material_path, stage=stage,
                    )


def configure_conveyor_motor(stage, speed_mps: float) -> None:
    """Drive the physical belt along its slider, leaving carried bodies to PhysX contacts."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    import isaaclab.sim as sim_utils

    speed_mps = float(speed_mps)
    if not math.isfinite(speed_mps) or speed_mps <= 0:
        raise ValueError(f"Invalid conveyor speed: {speed_mps}")
    stage = stage.stage if hasattr(stage, "stage") else stage
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        path = "/World/ConveyorBeltPhysics"
        joint_path = "/World/ConveyorBeltSlider"
        body = stage.GetPrimAtPath(path)
        if not body:
            low, high = _belt_range(stage)
            center = tuple(float((low[i] + high[i]) / 2) for i in range(3))
            size = tuple(float(high[i] - low[i]) for i in range(3))
            cfg = sim_utils.CuboidCfg(
                size=size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.6, dynamic_friction=0.5, restitution=0.0,
                ),
            )
            with sim_utils.use_stage(stage):
                body = cfg.func(path, cfg, translation=center)
            geometry = stage.GetPrimAtPath(f"{path}/geometry/mesh")
            UsdGeom.Imageable(geometry).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
            joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
            joint.CreateBody1Rel().SetTargets([Sdf.Path(path)])
            joint.CreateAxisAttr(UsdPhysics.Tokens.y)
            joint.CreateLocalPos0Attr(Gf.Vec3f(*center))
            joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            joint.CreateLowerLimitAttr(-1.0)
            joint.CreateUpperLimitAttr(1.0)
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
            drive.CreateStiffnessAttr(0.0)
            drive.CreateDampingAttr(100000.0)
            drive.CreateMaxForceAttr(100000.0)
        else:
            rigid_body = UsdPhysics.RigidBodyAPI(body)
            collider = stage.GetPrimAtPath(f"{path}/geometry/mesh")
            collision = UsdPhysics.CollisionAPI(collider)
            if (not rigid_body or not rigid_body.GetRigidBodyEnabledAttr().Get()
                    or rigid_body.GetKinematicEnabledAttr().Get()):
                raise ValueError(f"Invalid dynamic conveyor physics body at {path}")
            if not collision or not collision.GetCollisionEnabledAttr().Get():
                raise ValueError(f"Missing conveyor collider at {collider.GetPath()}")
        joint = UsdPhysics.PrismaticJoint.Get(stage, joint_path)
        if not joint or joint.GetBody1Rel().GetTargets() != [Sdf.Path(path)]:
            raise ValueError(f"Missing conveyor slider at {joint_path}")
        UsdPhysics.DriveAPI.Get(joint.GetPrim(), "linear").GetTargetVelocityAttr().Set(-speed_mps)


def sample_dynamic_reset(task: dict, seed: int) -> dict:
    """Sample the task's authored belt speed and placement ranges reproducibly."""
    from benchmark.scene.reset import _sample_range

    episode = task["episode_config"]
    rng = random.Random(seed)

    def uniform(bounds):
        return _sample_range(rng, bounds, context="TacDynamic reset")

    def xy(bounds):
        return tuple(uniform(bounds[axis]) for axis in ("x", "y"))

    reset = {
        "belt_speed_mps": uniform(episode["belt_speed_range_mps"]),
        "platform_x_offset_m": uniform(episode["platform_x_offset_range_m"]),
        "platform_entry_phase_m": uniform(episode["platform_entry_phase_range_m"]),
        "moving_asset_xy_jitter_m": tuple(
            uniform(episode.get("moving_asset_xy_jitter_range_m", {}).get(axis, (0, 0)))
            for axis in ("x", "y")
        ),
    }
    if episode["profile"] == "table_to_platform":
        bounds = episode.get("payload_position_range_m", episode.get("gear_position_range_m"))
        reset["table_payload_xy"] = xy(bounds) if bounds else None
    else:
        bounds = episode.get("fixture_position_range_m", episode.get("base_position_range_m"))
        reset["table_fixture_xy"] = xy(bounds) if bounds else None
    return reset


def settle_dynamic_scene(env, app, action):
    """Resolve thin source supports during landing, retaining the rollout clock."""
    from benchmark.backends.gsmini.execution import hold

    factor = env.cfg.dynamic_settle_substeps
    if factor == 1:
        hold(env, app, action, 1.8)
        return
    dt, decimation = env.cfg.sim.dt, env.cfg.decimation
    render_interval = env.cfg.sim.render_interval
    try:
        env.cfg.sim.dt = dt / factor
        env.cfg.decimation = decimation * factor
        env.cfg.sim.render_interval = render_interval * factor
        env.sim.set_simulation_dt(physics_dt=dt / factor, rendering_dt=dt * render_interval)
        hold(env, app, action, 1.8)
    finally:
        env.cfg.sim.dt, env.cfg.decimation = dt, decimation
        env.cfg.sim.render_interval = render_interval
        env.sim.set_simulation_dt(physics_dt=dt, rendering_dt=dt * render_interval)


def bind_dynamic_reset(env, task, reset, paths):
    """Retain authored initial states, never an episode's final physical poses."""
    env.dynamic_reset_context = (task, reset, paths, {
        name: asset.data.default_root_state.clone()
        for name, asset in env.scene.rigid_objects.items()
    })


def prepare_dynamic_reset(env, seed):
    """Resample the original layout before native reset of bodies and sensors."""
    task, initial, paths, states = env.dynamic_reset_context
    reset = sample_dynamic_reset(task, seed)
    roles = dynamic_scene_roles(task)
    platform_delta = (
        reset["platform_x_offset_m"] - initial["platform_x_offset_m"],
        initial["platform_entry_phase_m"] - reset["platform_entry_phase_m"], 0.0)
    moving_delta = tuple(platform_delta[i] + (
        reset["moving_asset_xy_jitter_m"][i] - initial["moving_asset_xy_jitter_m"][i]
        if i < 2 else 0.0) for i in range(3))
    table_key = "table_payload_xy" if roles["tabletop"] else "table_fixture_xy"
    table_delta = (tuple(reset[table_key][i] - initial[table_key][i] for i in range(2)) + (0.0,)
                   if reset.get(table_key) is not None else (0.0, 0.0, 0.0))
    for name, state in states.items():
        delta = ((0.0, 0.0, 0.0) if name == "conveyor" else
                 platform_delta if name == "platform" else
                 moving_delta if name == roles["moving"] or (
                     name == "source_holder" and roles["moving"] == "object") else table_delta)
        asset = env.scene[name]
        asset.data.default_root_state.copy_(state)
        asset.data.default_root_state[:, :3] += state.new_tensor(delta)
    # Jitter moves a mounted holder relative to the carrier, not just in world
    # space. Reset its permanent joint anchor too, or PhysX pulls it back toward
    # the previous seed and can eject the free payload on the first step.
    if task["episode_config"].get("carrier_mount") == "bolted_source_holder":
        from isaaclab.utils.math import quat_apply_inverse
        from pxr import Gf, Usd, UsdPhysics

        platform = env.scene["platform"].data.default_root_state
        holder = env.scene["source_holder"].data.default_root_state
        anchor = quat_apply_inverse(platform[:, 3:7], holder[:, :3] - platform[:, :3])
        anchor /= platform.new_tensor(task["episode_config"]["platform_size_m"])
        with Usd.EditContext(env.sim.stage, env.sim.stage.GetSessionLayer()):
            joint = UsdPhysics.FixedJoint.Get(env.sim.stage, PLATFORM_SOURCE_HOLDER_JOINT_PATH)
            joint.GetLocalPos0Attr().Set(Gf.Vec3f(*anchor[0].tolist()))
    configure_conveyor_motor(env.sim.stage, reset["belt_speed_mps"])
    env.cfg.seed = seed
    env.fixture_pose_history = env.fixture_velocity_y = None
    env.episode_start_step = None
    return reset, paths


def _table_top_z(stage) -> float:
    from pxr import Usd, UsdGeom, UsdPhysics

    table = stage.GetPrimAtPath("/World/BigTable")
    if not table:
        raise ValueError("Missing physical BigTable support")
    bounds = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy,
         UsdGeom.Tokens.guide],
        useExtentsHint=True,
    )
    tops = [float(bounds.ComputeWorldBound(prim).ComputeAlignedRange().GetMax()[2])
            for prim in Usd.PrimRange(table, Usd.TraverseInstanceProxies())
            if prim.HasAPI(UsdPhysics.CollisionAPI)
            and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()]
    if not tops:
        raise ValueError("BigTable has no enabled collision surface")
    return max(tops)


def _place_table_asset(stage, task: dict, reset: dict, role: str, path: str, *,
                       asset_type: str | None = None, scale=None, yaw: float | None = None,
                       position=None, dynamic: bool = True,
                       appearance: str | None = None) -> tuple[float, ...]:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from benchmark.catalog import object_asset
    from benchmark.scene.visuals import apply_dynamic_material

    episode = task["episode_config"]
    asset_type = asset_type or episode[
        "payload_asset_type" if role == "object" else "fixture_asset_type"]
    asset_path = object_asset(asset_type)
    asset_stage = Usd.Stage.Open(str(asset_path))
    root = asset_stage.GetDefaultPrim() if asset_stage else None
    if not root:
        raise ValueError(f"USD asset has no default prim: {asset_path}")
    scale = dynamic_asset_scale(task, role) if scale is None else tuple(map(float, scale))
    yaw = float(yaw if yaw is not None else episode.get(
        "base_yaw_deg" if role == "object" else "fixture_yaw_deg",
        episode.get("base_yaw_deg", 0.0)))
    if position is None:
        bounds = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render,
                                    UsdGeom.Tokens.proxy], useExtentsHint=True,
        )
        minimum = bounds.ComputeWorldBound(root).ComputeAlignedRange().GetMin()
        xy = reset["table_payload_xy" if role == "object" else "table_fixture_xy"]
        clearance = float(episode.get("base_mount_clearance_m", 0.002))
        position = (float(xy[0]), float(xy[1]),
                    _table_top_z(stage) - float(minimum[2]) * scale[2] + clearance)
    else:
        position = tuple(map(float, position))

    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    xform.AddRotateZOp().Set(yaw)
    xform.AddScaleOp().Set(Gf.Vec3d(*scale))
    asset_prim = UsdGeom.Xform.Define(stage, f"{path}/Asset")
    asset_prim.GetPrim().GetReferences().AddReference(str(asset_path.resolve()))
    if dynamic:
        mass = episode.get("payload_mass_kg" if role == "object" else "fixture_mass_kg")
        configure_moving_asset(stage, path, mass)
    elif asset_type in {"test_tube_base", "test_tube_hole_slot"}:
        # Kinematic fixtures support exact triangle meshes; keep payloads on SDF.
        # This avoids unstable GPU SDF/SDF settling without simplifying the bore or tip.
        body = _rigid_body_prim(stage, path)
        if not UsdPhysics.RigidBodyAPI(body).GetKinematicEnabledAttr().Get():
            raise ValueError(f"Triangle-mesh fixture must be kinematic: {path}")
        for prim in Usd.PrimRange(xform.GetPrim(), Usd.TraverseInstanceProxies()):
            if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")

    apply_dynamic_material(stage, path, task, role,
                           asset_type=asset_type, appearance=appearance)
    return position


def _place_table_source_assembly(stage, task: dict, reset: dict,
                                 holder_path: str, payload_path: str) -> None:
    episode = task["episode_config"]
    holder_type = episode["source_holder_type"]
    holder_scale = float(episode.get("source_holder_scale", 1.0))
    yaw = float(episode.get("base_yaw_deg", 0.0))
    instances = episode.get("source_fixture_instances") or episode.get("payload_instances", ())
    appearance = instances[0].get("material") if instances else None
    holder_position = _place_table_asset(
        stage, task, reset, "object", holder_path,
        asset_type=holder_type, scale=(holder_scale,) * 3,
        yaw=yaw, dynamic=False, appearance=appearance,
    )
    configure_source_holder_overrides(stage, holder_path, task)
    payload_position = dynamic_source_payload_position(task, holder_position)
    _place_table_asset(stage, task, reset, "object", payload_path,
                       yaw=yaw, position=payload_position)


def _rigid_body_prim(stage, path: str):
    from pxr import Usd, UsdPhysics

    prim = stage.GetPrimAtPath(path)
    bodies = [item for item in Usd.PrimRange(prim, Usd.TraverseInstanceProxies())
              if item.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(bodies) != 1:
        raise ValueError(f"Expected one rigid body under {path}, found {len(bodies)}")
    return bodies[0]


def _create_fixed_joint(stage, joint_path: str, body0_path: str, body1_path: str, *,
                        collision_enabled: bool = False) -> None:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    body0 = _rigid_body_prim(stage, body0_path)
    body1 = _rigid_body_prim(stage, body1_path)
    transforms = UsdGeom.XformCache()
    body0_pose = transforms.GetLocalToWorldTransform(body0)
    body1_pose = transforms.GetLocalToWorldTransform(body1)
    body0_rotation = Gf.Transform(body0_pose).GetRotation().GetQuat()
    body1_rotation = Gf.Transform(body1_pose).GetRotation().GetQuat()

    anchor = body1_pose.ExtractTranslation()
    local_anchor = body0_pose.GetInverse().Transform(anchor)
    local_rotation = body0_rotation.GetInverse() * body1_rotation
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([body0.GetPath()])
        joint.CreateBody1Rel().SetTargets([body1.GetPath()])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(local_anchor))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(local_rotation))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
        joint.CreateJointEnabledAttr().Set(True)
        joint.CreateCollisionEnabledAttr().Set(collision_enabled)


def _rigid_root_pose(stage, path: str):
    return _world_pose(stage, _rigid_body_prim(stage, path))


def _world_pose(stage, prim_or_path):
    from pxr import Gf, UsdGeom

    prim = (stage.GetPrimAtPath(prim_or_path) if isinstance(prim_or_path, str)
            else prim_or_path)
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    position = transform.ExtractTranslation()
    rotation = Gf.Transform(transform).GetRotation().GetQuat()
    return (tuple(float(position[i]) for i in range(3)),
            (float(rotation.GetReal()),
             *(float(rotation.GetImaginary()[i]) for i in range(3))))


def _seating_metrics(stage, paths, poses=None) -> dict:
    """Measure insertion and separation using the actual collision meshes."""
    if not paths.get("source_holder"):
        return {}
    import numpy as np
    import trimesh
    from pxr import Usd, UsdGeom, UsdPhysics
    from benchmark.scene.collision_geometry import mesh_geometry

    cache = UsdGeom.XformCache()
    meshes = {}
    for role in ("object", "source_holder"):
        path = paths[role]
        position, quaternion = _rigid_root_pose(stage, path)
        authored = trimesh.transformations.quaternion_matrix(quaternion)
        authored[:3, 3] = position
        actual = authored
        if poses is not None:
            position, quaternion = poses[role]
            actual = trimesh.transformations.quaternion_matrix(quaternion)
            actual[:3, 3] = position
        parts = []
        for prim in Usd.PrimRange(stage.GetPrimAtPath(path), Usd.TraverseInstanceProxies()):
            if (not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI)
                    or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False):
                continue
            mesh = mesh_geometry(prim)
            mesh.apply_transform(np.linalg.inv(authored)
                                 @ np.asarray(cache.GetLocalToWorldTransform(prim)).T)
            mesh.apply_transform(actual)
            parts.append(mesh)
        if not parts:
            raise RuntimeError(f"Source assembly has no collision geometry: {path}")
        meshes[role] = trimesh.util.concatenate(parts)
    holder, payload = meshes["source_holder"], meshes["object"]
    if not holder.is_watertight or not holder.is_winding_consistent:
        raise RuntimeError("Source holder requires closed, consistently oriented collision geometry")
    # Holder-frame measurements also work when the carrier tilts under load.
    holder_pose = (poses["source_holder"] if poses is not None else
                   _rigid_root_pose(stage, paths["source_holder"]))
    frame = trimesh.transformations.quaternion_matrix(holder_pose[1])
    frame[:3, 3] = holder_pose[0]
    inverse_frame = np.linalg.inv(frame)
    for mesh in (holder, payload):
        mesh.apply_transform(inverse_frame)
    probes = np.concatenate((payload.vertices, payload.triangles_center))
    closest, distances, faces = trimesh.proximity.closest_point(holder, probes)
    # Nearest outward face normals avoid random ray retries at aligned edges.
    outward = np.einsum("ij,ij->i", probes - closest, holder.face_normals[faces])
    penetration = distances[outward < -1e-9]
    metrics = {"insertion_depth_m": float(holder.bounds[1, 2] - payload.bounds[0, 2]),
               "support_distance_m": float(distances.min()),
               "penetration_m": float(penetration.max()) if len(penetration) else 0.0}
    return metrics


def _verify_source_seating(stage, paths, poses=None) -> dict:
    """Require an inserted source payload with real collider support, not just a joint."""
    metrics = _seating_metrics(stage, paths, poses)
    if not metrics:
        return metrics
    if (not all(math.isfinite(value) for value in metrics.values())
            or metrics["insertion_depth_m"] < 0.002
            or metrics["support_distance_m"] > 0.001
            or metrics["penetration_m"] > 0.001):
        raise RuntimeError(f"Source payload is not seated in its holder: {metrics}")
    return metrics


def _verify_target_insertion(stage, paths, poses=None) -> dict:
    """Require a held payload to enter the target without mesh penetration."""
    from benchmark.tasks.success import GoalNotReached

    metrics = _seating_metrics(stage,
        {"object": paths["object"], "source_holder": paths["fixture"]},
        None if poses is None else {"object": poses["object"], "source_holder": poses["fixture"]})
    if not metrics or not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError(f"Payload has not physically entered its target: {metrics}")
    if metrics["insertion_depth_m"] < 0.002 or metrics["penetration_m"] > 0.001:
        raise GoalNotReached(f"Payload has not physically entered its target: {metrics}")
    return metrics


def _prepare_dynamic_scene(stage, task: dict, reset: dict, *,
                           include_held_payload: bool = True,
                           drop_clearance_m: float = 0.01) -> dict[str, str | None]:
    from pxr import Gf, Usd, UsdGeom
    from benchmark.catalog import object_asset
    from benchmark.scene.preview_scene import prepare_dynamic_preview
    from benchmark.scene.visuals import apply_dynamic_material

    episode = task["episode_config"]
    roles = dynamic_scene_roles(task)
    prepare_dynamic_preview(stage, task, drop_clearance_m=drop_clearance_m, reset=reset)
    moving_path = "/World/PreviewMovingAsset"
    table_payload_path = "/World/TacDynamicTablePayload"
    table_fixture_path = "/World/TacDynamicTableFixture"
    table_holder_path = "/World/TacDynamicTableSourceHolder"
    held_payload_path = "/World/TacDynamicHeldPayload"

    with Usd.EditContext(stage, stage.GetSessionLayer()):
        if roles["tabletop"] == "object":
            if roles["source_holder"]:
                _place_table_source_assembly(
                    stage, task, reset, table_holder_path, table_payload_path,
                )
            else:
                _place_table_asset(stage, task, reset, "object", table_payload_path)
        if roles["static_fixture"] == "fixture":
            _place_table_asset(stage, task, reset, "fixture", table_fixture_path,
                               dynamic=False)
        target_opening = (episode.get("target_fixture_opening_x_m"),
                          episode.get("target_fixture_opening_y_m"))
        if episode.get("fixture_asset_type") == "hdmi_slot" and any(
                value is not None for value in target_opening):
            target_path = (moving_path if roles["moving"] == "fixture"
                           else table_fixture_path)
            target_task = {"episode_config": {
                "source_holder_type": "hdmi_slot",
                "source_holder_scale": episode.get("fixture_scale", 1.0),
                "source_holder_opening_x_m": target_opening[0],
                "source_holder_opening_y_m": target_opening[1],
                "source_holder_wall_height_m": episode.get(
                    "target_fixture_wall_height_m"),
            }}
            configure_source_holder_overrides(stage, target_path, target_task)
        if (include_held_payload and roles["tabletop"] is None
                and roles["moving"] == "fixture"):
            asset_path = object_asset(episode["payload_asset_type"])
            held_xform = UsdGeom.Xform.Define(stage, held_payload_path)
            held_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, _table_top_z(stage) + 0.25))
            held_xform.AddScaleOp().Set(Gf.Vec3d(*dynamic_asset_scale(task, "object")))
            held_asset = UsdGeom.Xform.Define(stage, f"{held_payload_path}/Asset")
            held_asset.GetPrim().GetReferences().AddReference(str(asset_path.resolve()))
            configure_moving_asset(stage, held_payload_path, episode.get("payload_mass_kg"))
            apply_dynamic_material(stage, held_payload_path, task, "object")
        if (roles["moving"] == "fixture"
                and episode.get("carrier_mount") == "bolted_fixture"):
            _create_fixed_joint(
                stage, PLATFORM_FIXTURE_JOINT_PATH, PLATFORM_PATH, moving_path,
            )
        if (roles["moving"] == "object" and roles["source_holder"]
                and episode.get("carrier_mount") == "bolted_source_holder"):
            _create_fixed_joint(
                stage, PLATFORM_SOURCE_HOLDER_JOINT_PATH, PLATFORM_PATH,
                "/World/PreviewSourceHolder",
            )

    platform_pose, _ = _rigid_root_pose(stage, PLATFORM_PATH)
    robot_pose = _world_pose(stage, "/World/FrankaArm")
    if roles["moving"] == "fixture":
        fixture_path = moving_path
        object_path = held_payload_path if roles["tabletop"] is None else table_payload_path
    else:
        object_path, fixture_path = moving_path, table_fixture_path
    paths = {
        "object": object_path,
        "fixture": fixture_path,
        "platform": PLATFORM_PATH,
        "source_holder": ("/World/PreviewSourceHolder" if roles["moving"] == "object"
                           else table_holder_path) if roles["source_holder"] else None,
        "robot": "/World/FrankaArm",
        "table": "/World/BigTable",
    }
    paths["_robot_pose"] = robot_pose
    paths["_platform_pose"] = platform_pose
    _verify_source_seating(stage, paths)
    return paths


def _dynamic_obstacle_paths(paths):
    return {
        "Table": paths["table"],
        "Conveyor": "/World/ConveyorBeltPhysics",
        **{f"{belt}_{side}": f"/World/Ur10Table/{belt}/{side}"
           for belt in ("conveyor", "conveyor_01")
           for side in ("left_wall", "right_wall")},
        **{name: paths[name] for name in ("object", "fixture", "platform", "source_holder")
           if paths[name] is not None},
    }


def _make_env_cfg(seed: int, task: dict, stage, paths: dict[str, str | None]):
    from benchmark.backends.gsmini.sensors import FrankaGsminiLiftEnvCfg, configure_contact_sensors
    from benchmark.backends.gsmini.environment import _configure_dynamic_env_cfg, _configure_in_hand_joints, _remap_robot_paths
    cfg = FrankaGsminiLiftEnvCfg()
    cfg.dynamic_settle_substeps = 4 if task["task_id"] == 4 else 1
    _configure_dynamic_env_cfg(cfg, seed, task, stage, paths, _rigid_root_pose)

    episode = task["episode_config"]
    if episode.get("in_hand_reset", {}).get("mode") == "in_hand":
        reset_task = {"reset_contract": episode["in_hand_reset"]}
        _configure_in_hand_joints(cfg.scene.robot.init_state.joint_pos, reset_task)
        if episode["object_family"] == "bulb":
            cfg.actions.gripper_action.close_command_expr = {
                "panda_finger_joint.*": float(episode["in_hand_reset"]["gripper_joint_pos"])}
    _remap_robot_paths(
        cfg.scene, ("ee_frame", "agentview", "wristview", "gsmini_left", "gsmini_right"),
        paths["robot"],
    )
    payload_body_path = str(_rigid_body_prim(stage, paths["object"]).GetPath())
    # RigidObject resolves descendants, while FOTS requires the body itself.
    configure_contact_sensors(cfg, [payload_body_path], robot_path=paths["robot"],
                              marker_target_path=payload_body_path)
    # This suite reuses an authored robot (spawn=None), so spawning cannot enable reports.
    from pxr import Usd
    from isaaclab.sim import activate_contact_sensors
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        activate_contact_sensors(paths["robot"], stage=stage)

    return cfg


def _load_dynamic_stage(app, task: dict):
    import isaaclab.sim as sim_utils
    from isaacsim.core.utils.stage import is_stage_loading
    from benchmark.catalog import scene_asset

    path = scene_asset(task)
    if not sim_utils.open_stage(str(path)):
        raise RuntimeError(f"Could not open TacDynamic scene: {path}")
    robot_path = "/World/FrankaArm"
    from benchmark.scene.visuals import expand_robot_visual_instances

    # Expand visual references before the first Hydra update, while physics
    # is stopped. The loaded GSmini payload's collision instances are retained.
    stage = sim_utils.get_current_stage()
    if stage.GetPrimAtPath(robot_path).IsValid():
        expand_robot_visual_instances(stage, robot_path)
    for _ in range(120):
        app.update()
        stage = sim_utils.get_current_stage()
        if is_stage_loading() or not stage.GetPrimAtPath(robot_path).IsValid():
            continue
        ready = all(stage.GetPrimAtPath(f"{robot_path}/{part}").IsValid()
                    for part in ("panda_link0/panda_joint1", "panda_hand"))
        if ready:
            # A delayed payload may only become available during app.update().
            expand_robot_visual_instances(stage, robot_path)
            configure_gsmini_pad_material(stage, robot_path)
            return stage
    raise RuntimeError(f"TacDynamic scene or robot payload failed to load: {path}")
