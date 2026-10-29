"""Native physics-step diagnostics for the five additional rigid contact tasks.

Only reset writes object state. This module observes contacts and motion; robot
actions remain the sole means of completing an episode.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation


CONTACT_KINDS = frozenset(("push", "roll", "gears", "hang", "slip"))


def aggregate_contact_segments(values, counts, starts, *, average, default):
    """Aggregate native contact ranges without host reads or dynamic CUDA sizes."""
    import torch

    indices = torch.arange(values.shape[0], device=values.device)
    mask = ((indices >= starts.reshape(-1, 1))
            & (indices < (starts+counts).reshape(-1, 1)))
    # Unpopulated buffer slots are unspecified and may contain NaN. Mask
    # before reduction: multiplying NaN by zero would still corrupt the sum.
    result = torch.where(mask[..., None], values[None], 0.).sum(dim=1)
    flat_counts = counts.reshape(-1, 1)
    if average:
        result = result/flat_counts.clamp_min(1)
    result = torch.where(flat_counts > 0, result, default)
    return result.reshape(*counts.shape, 3)


def target_contact_magnitude(values, counts, starts):
    """First-filter magnitude sum, masking unpopulated native slots on device."""
    import torch

    index = torch.arange(values.shape[0], device=values.device)
    start, count = starts.reshape(-1)[0], counts.reshape(-1)[0]
    populated = (index >= start) & (index < start+count)
    vectors = values.reshape(values.shape[0], math.prod(values.shape[1:]))
    return torch.linalg.vector_norm(torch.where(populated[:, None], vectors, 0.), dim=-1).sum()


def configure_contact_sensors(cfg, task):
    """Track support and transmission separately from robot normal/shear loads."""
    from isaaclab.sensors import ContactSensorCfg
    from benchmark.backends.gsmini.sensors import PhysicsStepContactSensor
    from pxr import Usd, UsdPhysics

    class ContactSensor(PhysicsStepContactSensor):
        def _unpack_contact_buffer_data(self, contact_data, buffer_count, buffer_start_indices,
                                        avg=True, default=float("nan")):
            if not avg:
                # Isaac calls this reduction for friction; retain the same
                # native patches for the slip mechanical-load upper bound.
                self._extra_contact_friction_data = (contact_data, buffer_count, buffer_start_indices)
            return aggregate_contact_segments(contact_data, buffer_count, buffer_start_indices,
                                              average=avg, default=default).reshape(
                self._num_envs, self._num_bodies, self.contact_physx_view.filter_count, 3)

    prefix = "{ENV_REGEX_NS}"
    kind = task["episode_config"]["task_type"]
    def native_paths(path, label, schema):
        stage = Usd.Stage.Open(path)
        root = stage.GetDefaultPrim().GetPath()
        paths = []
        for prim in stage.Traverse():
            if not prim.HasAPI(schema):
                continue
            if schema == UsdPhysics.CollisionAPI and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
                continue
            paths.append(f"{label}/{prim.GetPath().MakeRelativePath(root)}")
        if not paths:
            raise RuntimeError(f"No explicit native {label} contact targets in {path}")
        return paths
    robot_paths = native_paths(cfg.scene.robot.spawn.usd_path, "Robot", UsdPhysics.RigidBodyAPI)
    table_paths = native_paths(cfg.scene.table.spawn.usd_path, "Table", UsdPhysics.CollisionAPI)
    cfg.extra_contact_robot_filter_count = len(robot_paths)
    cfg.extra_contact_table_filter_count = len(table_paths)
    for side in ("left", "right"):
        for suffix in ("contact", "case_contact"):
            contact = getattr(cfg.scene, f"gsmini_{side}_{suffix}")
            contact.class_type = ContactSensor
            # These tasks consume raw native points only for reset penetration
            # and hook-region checks. Averaged points are otherwise unused.
            contact.track_contact_points = False
            contact.track_friction_forces = True
            contact.max_contact_data_count_per_prim = 128

    def sensor(name, body, filters):
        setattr(cfg.scene, name, ContactSensorCfg(
            class_type=ContactSensor, prim_path=f"{prefix}/{body}",
            update_period=0., history_length=cfg.decimation,
            filter_prim_paths_expr=[f"{prefix}/{item}" for item in filters],
            track_contact_points=False, track_friction_forces=True,
            max_contact_data_count_per_prim=256))

    if kind == "gears":
        sensor("gear_contact", "mechanism/Medium", ["mechanism/Small", "mechanism/Large"])
        for body in ("Small", "Medium", "Large"):
            others = [f"mechanism/{other}" for other in ("Small", "Medium", "Large") if other != body]
            sensor(f"audit_{body.lower()}_contact", f"mechanism/{body}", [*table_paths, *robot_paths, *others])
    else:
        body = task["objects"]["mechanism"]["contact_bodies"][0]
        sensor("support_contact", f"mechanism/{body}", table_paths)
        sensor("object_robot_contact", f"mechanism/{body}", robot_paths)
    if kind == "hang":
        sensor("hook_contact", "hook/Hook", ["mechanism/Mug"])


def rotation(wxyz):
    q = np.asarray(wxyz)
    return Rotation.from_quat(q[[1, 2, 3, 0]])


def rolling_contact_speed(linear_velocity, angular_velocity, contact_offset):
    """Material-point velocity, rather than an endpoint rolling approximation."""
    return np.asarray(linear_velocity) + np.cross(angular_velocity, contact_offset)


def unwrap_increment(previous, current):
    return math.atan2(math.sin(current-previous), math.cos(current-previous))


def draw_push_path(observer):
    """Draw the reset-relative goal as paint; it has no physical schemas."""
    from pxr import Gf, UsdGeom

    u = np.linspace(0., 1., 81)
    local = np.column_stack((observer.goal["length_m"]*u,
                             observer.goal["amplitude_m"]*np.sin(2*math.pi*u), np.zeros_like(u)))
    points = observer.layout.apply(local)+observer.position
    points[:, 2] = observer.table_height+.0005
    curve = UsdGeom.BasisCurves.Define(observer.env.sim.stage, "/World/envs/env_0/PushPath")
    curve.CreateTypeAttr("linear")
    curve.CreateCurveVertexCountsAttr([len(points)])
    curve.CreatePointsAttr([Gf.Vec3f(*[float(v) for v in p]) for p in points])
    curve.CreateWidthsAttr([.0015])
    curve.SetWidthsInterpolation("constant")
    curve.CreateDisplayColorAttr([Gf.Vec3f(.02, .55, 1.)])


def reset_fixed_mounts(env, poses):
    """Move a fixed fixture's world attachment together with its seeded root."""
    from pxr import Gf, Usd, UsdPhysics

    if not env._resetting:
        raise RuntimeError("Fixed mount frames may only change during reset")
    stage = env.sim.stage
    for name, spec in env.task["objects"].items():
        if not spec.get("fixed_mount"):
            continue
        pose = poses[name]
        root_rotation = rotation(pose["quaternion_wxyz"])
        root_position = np.asarray(pose["position"])+env.scene.env_origins[0].cpu().numpy()
        root = stage.GetPrimAtPath(f"/World/envs/env_0/{name}")
        mounts = []
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdPhysics.FixedJoint):
                continue
            joint = UsdPhysics.FixedJoint(prim)
            if joint.GetBody0Rel().GetTargets() or not joint.GetBody1Rel().GetTargets():
                continue
            p1 = np.asarray(joint.GetLocalPos1Attr().Get())
            q1 = joint.GetLocalRot1Attr().Get()
            child_rotation = Rotation.from_quat([*q1.GetImaginary(), q1.GetReal()])
            p0 = root_position+root_rotation.apply(p1)
            q0 = (root_rotation*child_rotation).as_quat()
            joint.GetLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in p0]))
            joint.GetLocalRot0Attr().Set(Gf.Quatf(float(q0[3]), Gf.Vec3f(*[float(v) for v in q0[:3]])))
            mounts.append(str(prim.GetPath()))
        if not mounts:
            raise RuntimeError(f"No world-fixed joint found for the fixed fixture {name}")


def native_initial_state(env):
    """Snapshot native roots, twists, joints and the robot command at reset."""
    states = {}
    for name in ("robot", *env.task["objects"]):
        asset = env.scene[name]
        row = {"root_state": asset.data.root_state_w.cpu().tolist()}
        if hasattr(asset.data, "joint_pos"):
            row.update(joint_positions=asset.data.joint_pos.cpu().tolist(),
                       joint_velocities=asset.data.joint_vel.cpu().tolist())
        states[name] = row
    return {"assets": states, "action": env.action.detach().cpu().tolist()}


def restore_native_initial_state(env, state):
    """Restore a recorded native reset before the first robot action."""
    import torch
    from benchmark.tasks.extra.environment import ExtraProgress

    if env.extra_step_counter != 0:
        raise RuntimeError("Native initialization may only be restored before episode actions")
    if set(state["assets"]) != {"robot", *env.task["objects"]}:
        raise ValueError("Replay native initialization has different task bodies")
    env._resetting = True
    runtime = getattr(env, "contact_runtime", None)
    if runtime is not None:
        runtime.initialized = False
    try:
        for name, row in state["assets"].items():
            asset = env.scene[name]
            value = torch.tensor(row["root_state"], device=env.device, dtype=torch.float32)
            asset.write_root_pose_to_sim(value[:, :7])
            asset.write_root_velocity_to_sim(value[:, 7:])
            if "joint_positions" in row:
                q = torch.tensor(row["joint_positions"], device=env.device, dtype=torch.float32)
                dq = torch.tensor(row["joint_velocities"], device=env.device, dtype=torch.float32)
                asset.write_joint_state_to_sim(q, dq)
                if name == "robot":
                    asset.set_joint_position_target(q)
        env.action.copy_(torch.tensor(state["action"], device=env.device, dtype=env.action.dtype))
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(0.)
        env.scene.reset()
        env.progress = ExtraProgress(env.task)
        env.initial_state = env._snapshot()
        if runtime is not None:
            runtime.reset()
    finally:
        env._resetting = False


class ContactRuntime:
    """One post-update observer, also consumed by recording and validation."""

    def __init__(self, env):
        self.env = env
        self.kind = env.task["episode_config"]["task_type"]
        self.spec = env.task["objects"]["mechanism"]
        from benchmark.tasks.extra.contact_assets import asset_specs
        self.geometry = asset_specs()[self.spec["type"]]
        self.hook_geometry = asset_specs().get("extra_mug_hook")
        self.goal = env.task["goals"][0]
        self.initialized = False
        self.last_sample = {}
        self._install_observer()

    def _install_observer(self):
        """Sample after native scene updates; skip reset and zero-time refreshes."""
        env = self.env
        self.original_update = env.scene.update

        def update(dt, *args, **kwargs):
            result = self.original_update(dt, *args, **kwargs)
            if dt > 0 and self.initialized and not env._resetting:
                self.physics_steps += 1
                self.last_sample = self.sample(float(dt))
                env.progress.update(self.last_sample, float(dt), sample_id=self.physics_steps)
            return result

        env.scene.update = update

    def pose(self, name="mechanism"):
        cached = getattr(self, "_cpu_cache", {}).get(f"{name}/root")
        if cached is not None:
            return cached[:3].copy(), rotation(cached[3:7])
        data = self.env.scene[name].data
        return data.root_pos_w[0].detach().cpu().numpy().copy(), rotation(
            data.root_quat_w[0].detach().cpu().numpy())

    def hand_pose(self):
        cached = getattr(self, "_cpu_cache", {}).get("robot/hand")
        if cached is not None:
            return cached[:3].copy(), rotation(cached[3:7])
        robot = self.env.scene["robot"]
        index = robot.body_names.index("panda_hand")
        return robot.data.body_pos_w[0, index].detach().cpu().numpy().copy(), rotation(
            robot.data.body_quat_w[0, index].detach().cpu().numpy())

    def force(self, name):
        """Normal and tangential vectors acting on the sensor body, by filter."""
        cached = getattr(self, "_cpu_cache", {})
        if f"{name}/normal" in cached:
            normal = cached[f"{name}/normal"].reshape(-1, 3)
            tangent = cached[f"{name}/tangent"].reshape(-1, 3)
        else:
            data = self.env.scene[name].data
            normal = data.force_matrix_w[0].detach().cpu().numpy().reshape(-1, 3)
            tangent = data.friction_forces_w[0].detach().cpu().numpy().reshape(-1, 3)
        sensor = self.env.scene[name]
        if len(normal) != len(sensor.cfg.filter_prim_paths_expr):
            raise RuntimeError(f"Native contact filter count differs from explicit targets: {name}")
        if not np.isfinite(normal).all() or not np.isfinite(tangent).all():
            raise RuntimeError(f"Nonfinite native contact force: {name}")
        return normal, tangent

    def reset(self):
        self.initialized = False
        self.physics_steps = 0
        self.position, self.orientation = self.pose()
        self.layout = rotation(self.env.reset_poses["mechanism"]["quaternion_wxyz"])
        self.roll_angle = self.previous_roll = self.slip_distance = self.arc_length = 0.
        self.roll_tracking = False
        self.gear_angles = np.zeros(3)
        if self.kind == "gears":
            asset = self.env.scene["mechanism"]
            self.gear_indices = [asset.joint_names.index(name) for name in
                                 ("medium_turn", "small_turn", "large_turn")]
            self.previous_gears = asset.data.joint_pos[0, self.gear_indices].cpu().numpy().copy()
            requested = self.env.reset_poses["mechanism"]["joint_positions"]
            targets = np.array([requested[asset.joint_names[index]] for index in self.gear_indices])
            # Use the history's existing 0.02 rad start tolerance before
            # rebasing: settling must not hide a collision-driven large turn.
            error = self.previous_gears-targets
            if not np.isfinite(error).all() or (abs(error) > .02).any():
                raise RuntimeError("Gear joints did not retain their seeded reset positions")
        self.slip_reference = None
        self.arm_stable_s = 0.
        self.previous_hook_side = None
        self.threaded = False
        self.threading_height = None
        self.previous_axial = 0.
        self.history = []
        self.table_height = self._table_height()
        if self.kind == "push" and hasattr(self.env, "sim"):
            draw_push_path(self)
        self.initialized = True
        self.last_sample = self.sample(0.)

    def _table_height(self):
        import omni.physx

        values = []
        def hit(item):
            if "/Table/" in item.collision:
                values.append(float(item.position[2]))
            return True
        omni.physx.get_physx_scene_query_interface().raycast_all(
            (float(self.position[0]), float(self.position[1]), 1.), (0., 0., -1.), 1., hit)
        if not values:
            raise RuntimeError("No physical tabletop below the new contact task")
        return max(values)

    def lowest_point(self, position, orientation):
        from itertools import product

        corners = np.asarray(list(product(*zip(*self.spec["view_bounds_local_m"]))))
        return float((orientation.apply(corners)+position)[:, 2].min())

    def _snapshot_tensors(self):
        """One synchronized native snapshot, avoiding a GPU wait per scalar."""
        import torch

        tensors = {}
        def root(name):
            data = self.env.scene[name].data
            tensors[f"{name}/root"] = torch.cat((data.root_pos_w[0], data.root_quat_w[0],
                                                  data.root_lin_vel_w[0], data.root_ang_vel_w[0]))
        root("mechanism")
        names = [f"gsmini_{side}_{suffix}" for side in ("left", "right")
                 for suffix in ("contact", "case_contact")]
        if self.kind == "gears":
            names += ["gear_contact", *(f"audit_{body}_contact" for body in ("small", "medium", "large"))]
            data = self.env.scene["mechanism"].data
            tensors["gears/position"] = data.joint_pos[0, self.gear_indices]
            tensors["gears/velocity"] = data.joint_vel[0, self.gear_indices]
        else:
            names += ["support_contact", "object_robot_contact"]
        if self.kind == "hang":
            names.append("hook_contact")
            root("hook")
            robot = self.env.scene["robot"]
            tensors["robot/body_positions"] = robot.data.body_pos_w[0]
            tensors["robot/body_quaternions"] = robot.data.body_quat_w[0]
        if self.kind == "slip":
            robot = self.env.scene["robot"]
            i = robot.body_names.index("panda_hand")
            data = robot.data
            tensors["robot/hand"] = torch.cat((data.body_pos_w[0, i], data.body_quat_w[0, i],
                                                data.body_lin_vel_w[0, i], data.body_ang_vel_w[0, i]))
            finger_indices = [robot.joint_names.index(name) for name in
                              ("panda_finger_joint1", "panda_finger_joint2")]
            tensors["robot/gripper_aperture"] = data.joint_pos[0, finger_indices].sum()
            tensors["robot/gripper_command"] = self.env.action[0, -1]
        for name in names:
            data = self.env.scene[name].data
            tensors[f"{name}/normal"] = data.force_matrix_w[0]
            tensors[f"{name}/tangent"] = data.friction_forces_w[0]
        if self.kind == "slip":
            loads = []
            for side in ("left", "right"):
                sensor = self.env.scene[f"gsmini_{side}_case_contact"]
                normal, _, _, _, counts, starts = sensor.contact_physx_view.get_contact_data(
                    dt=self.env.cfg.sim.dt)
                load = target_contact_magnitude(normal, counts, starts)
                friction = getattr(sensor, "_extra_contact_friction_data", None)
                if friction is None:
                    # Native adapters without the retained Isaac friction
                    # reduction still use the same raw view and timestep.
                    force, _, counts, starts = sensor.contact_physx_view.get_friction_data(dt=self.env.cfg.sim.dt)
                    friction = force, counts, starts
                loads.append(load+target_contact_magnitude(*friction))
            # Normal contacts and friction patches are not one-to-one. The
            # sum of their magnitudes is an upper bound, never gel support.
            # Pack the device scalar with the other native data: no extra
            # per-side host copies or a second friction-buffer query.
            tensors["robot/housing_contact_load"] = sum(loads)
        tensors.update(self._extra_snapshot_tensors())
        packed = torch.cat([value.reshape(-1) for value in tensors.values()]).detach().cpu().numpy()
        result, offset = {}, 0
        for key, value in tensors.items():
            count = value.numel()
            result[key] = packed[offset:offset+count].reshape(tuple(value.shape))
            offset += count
        return result

    def _extra_snapshot_tensors(self):
        return {}

    def sample(self, dt):
        self._cpu_cache = self._snapshot_tensors()
        try:
            return self._sample(dt)
        finally:
            self._cpu_cache = {}

    def _sample(self, dt):
        env, kind = self.env, self.kind
        p, r = self.pose()
        native = self._cpu_cache["mechanism/root"]
        velocity, omega = native[7:10], native[10:13]
        state = {"object_speed_m_s": float(np.linalg.norm(velocity)),
                 "angular_speed_rad_s": float(np.linalg.norm(omega)),
                 "object_position_w_m": p.tolist(),
                 "object_quaternion_wxyz": native[3:7].tolist()}
        gel_up = housing_up = 0.
        for part, suffix in (("gel", "contact"), ("housing", "case_contact")):
            vectors, normals, magnitudes, all_normal = [], [], [], []
            for side in ("left", "right"):
                normal, tangent = self.force(f"gsmini_{side}_{suffix}")
                # The first filter is the manipulated body. Touching the hook
                # or an output gear must not count as input-pad interaction.
                vectors.append(-(normal+tangent)[:1].sum(axis=0))
                normals.append(float(np.linalg.norm(normal[:1], axis=-1).sum()))
                all_normal.append(float(np.linalg.norm(normal, axis=-1).sum()))
                magnitudes.append(float(np.linalg.norm((normal+tangent)[:1], axis=-1).sum()))
                if part == "gel":
                    state[f"{side}_gel_force_n"] = normals[-1]
                state[f"{side}_{part}_normal_force_w_n"] = normal[:1].sum(axis=0).tolist()
                state[f"{side}_{part}_tangent_force_w_n"] = tangent[:1].sum(axis=0).tolist()
            state[f"{part}_force_n"] = sum(normals)
            state[f"all_{part}_force_n"] = sum(all_normal)
            state[f"{part}_total_force_n"] = sum(magnitudes)
            net_up = max(0., sum(v[2] for v in vectors))
            positive_up = sum(max(0., v[2]) for v in vectors)
            if part == "gel":
                # Gross positive reactions cannot certify effective weight
                # support when the opposite pad pushes the object down.
                gel_up = net_up
                state["gel_positive_up_force_n"] = positive_up
            else:
                # A downward reaction on one housing must not hide upward
                # support on the other. Within each native body/filter this
                # remains a resultant: opposing contacts on one housing can
                # still cancel, so this is not a per-contact load bound.
                housing_up = positive_up
                state["housing_net_up_force_n"] = net_up
        state.update(robot_force_n=state["gel_total_force_n"]+state["housing_total_force_n"],
                     gel_up_force_n=gel_up, housing_up_force_n=housing_up)
        if kind != "gears":
            support_n, support_t = self.force("support_contact")
            support = support_n+support_t
            state["table_force_n"] = float(np.linalg.norm(support, axis=-1).sum())
            normal, tangent = self.force("object_robot_contact")
            state["robot_force_n"] = float(np.linalg.norm(normal+tangent, axis=-1).sum())
        if kind == "push":
            local = self.layout.inv().apply(p-self.position)
            length, amplitude = self.goal["length_m"], self.goal["amplitude_m"]
            u = float(local[0]/length)
            state.update(path_u=u, cross_track_m=float(abs(local[1]-amplitude*math.sin(2*math.pi*u))),
                         lift_m=float(local[2]))
        elif kind == "roll":
            from benchmark.tasks.extra.contact import cylinder_minimum_z

            relative = self.orientation.inv()*r
            raw = math.atan2(relative.as_matrix()[0, 2], relative.as_matrix()[2, 2])
            increment = unwrap_increment(self.previous_roll, raw)
            self.previous_roll = raw
            self.roll_angle += increment
            radius = self.goal["radius_m"]
            axle = r.apply([0., 1., 0.])
            tangent_velocity = rolling_contact_speed(velocity, omega, [0., 0., -radius])
            translation = float(self.layout.inv().apply(p-self.position)[0])
            # Settling drift and noise-level grazes before a real preload are
            # not manipulation; arming there divides by a near-zero arc.
            started = (max(abs(self.roll_angle)*radius,float(np.linalg.norm((p-self.position)[:2]))) > .0001
                       and state["gel_force_n"] >= self.env.task["success_config"]["contact_threshold_n"])
            if self.roll_tracking:
                self.slip_distance += float(np.linalg.norm(tangent_velocity[:2]))*dt
                self.arc_length += abs(increment)*radius
            self.roll_tracking |= started
            state.update(roll_angle_rad=self.roll_angle,
                         translation_m=translation,
                         cross_track_m=float(abs(self.layout.inv().apply(p-self.position)[1])),
                         slip_ratio=self.slip_distance/max(self.arc_length, 1e-6),
                         rolling_slip_distance_m=self.slip_distance, rolling_arc_length_m=self.arc_length,
                         off_axis_rad=math.acos(float(np.clip(np.dot(axle, self.orientation.apply([0., 1., 0.])), -1., 1.))),
                         lift_m=float(p[2]-self.position[2]),
                         table_clearance_m=cylinder_minimum_z(float(p[2]), float(axle[2]), radius,
                                                              self.geometry["length_m"])-self.table_height)
        elif kind == "gears":
            current = self._cpu_cache["gears/position"]
            self.gear_angles += [unwrap_increment(a, b) for a, b in zip(self.previous_gears, current)]
            self.previous_gears = current.copy()
            normal, tangent = self.force("gear_contact")
            pairs = np.linalg.norm(normal+tangent, axis=-1)
            output_force = 0.
            first = env.cfg.extra_contact_table_filter_count
            last = first+env.cfg.extra_contact_robot_filter_count
            n, t = self.force("audit_medium_contact")
            input_force = float(np.linalg.norm((n+t)[first:last], axis=-1).sum())
            for body in ("small", "large"):
                n, t = self.force(f"audit_{body}_contact")
                output_force += float(np.linalg.norm((n+t)[first:last], axis=-1).sum())
            state.update(input_angle_rad=float(self.gear_angles[0]),
                         output_fast_angle_rad=float(self.gear_angles[1]),
                         output_slow_angle_rad=float(self.gear_angles[2]),
                         pair_fast_force_n=float(pairs[0]), pair_slow_force_n=float(pairs[1]),
                         robot_output_force_n=output_force,
                         other_robot_input_force_n=max(0., input_force-state["gel_total_force_n"]
                                                       -state["housing_total_force_n"]),
                         robot_force_n=input_force+output_force,
                         angular_speed_rad_s=float(np.abs(self._cpu_cache["gears/velocity"]).max()))
            for side in ("left", "right"):
                present = state[f"{side}_gel_force_n"] >= .05
                count, checked = 0, False
                if not present:
                    sensor = self.env.scene[f"gsmini_{side}_contact"]
                    *_, counts, _ = sensor.contact_physx_view.get_contact_data(dt=self.env.cfg.sim.dt)
                    counts = counts.detach().cpu().numpy().reshape(-1)
                    if len(counts) != len(sensor.cfg.filter_prim_paths_expr) or (counts < 0).any():
                        raise RuntimeError("Gear native contact counts do not match target filters")
                    count, checked = int(counts[0]), True
                    reaction = max(np.linalg.norm(state[f"{side}_gel_{component}_force_w_n"])
                                   for component in ("normal", "tangent"))
                    present = count > 0 and reaction > 1e-6
                state.update({f"{side}_gel_contact_present": bool(present),
                              f"{side}_gel_low_force_native_contact_count": count,
                              f"{side}_gel_contact_count_checked": checked})
        elif kind == "hang":
            self._hang_state(state, p, r)
        elif kind == "slip":
            self._slip_state(state, p, r, dt)
        if dt > 0:
            self.history.append(state)
            del self.history[:-self.env.cfg.decimation]
        return state

    def _hang_state(self, state, p, r):
        spec = self.geometry
        hp, hr = self.pose("hook")
        hook_spec = self.hook_geometry
        tip = hp+hr.apply(hook_spec["tip_local_m"])
        seat = hp+hr.apply(hook_spec["seat_local_m"])
        center = np.asarray(spec["handle_center_local_m"])
        relative = r.inv().apply(tip-p)-center
        inner_rx, inner_rz = spec["handle_inner_ellipse_radii_xz_m"]
        inside = (relative[0]/inner_rx)**2+(relative[2]/inner_rz)**2 < .9**2
        if (inside and self.previous_hook_side is not None and self.previous_hook_side*relative[1] <= 0
                and abs(relative[1]-self.previous_hook_side) < .01):
            self.threaded = True
            if self.threading_height is None:
                self.threading_height = float((p+r.apply(center))[2])
        self.previous_hook_side = float(relative[1])
        normal, tangent = self.force("hook_contact")
        upward = max(0., float(-(normal+tangent).sum(axis=0)[2]))
        valid, _ = self.contact_points("hook_contact") if upward > 1e-5 else (np.empty((0, 3)), None)
        handle_only = bool(len(valid))
        if len(valid):
            local = r.inv().apply(valid-p)-center
            theta = np.linspace(0., 2*math.pi, 512, endpoint=False)
            tube_rx, tube_rz = spec["handle_ellipse_radii_m"]
            curve = np.column_stack((tube_rx*np.cos(theta), np.zeros_like(theta), tube_rz*np.sin(theta)))
            distance = np.linalg.norm(local[:, None, :]-curve[None, :, :], axis=-1).min(axis=1)
            cup_points = r.inv().apply(valid-p)
            cup_points[:, 0] -= spec["body_center_x_m"]
            outside_body = np.linalg.norm(cup_points[:, :2], axis=-1) > spec["body_radius_m"]+.0005
            outside_body |= ((cup_points[:, 2] < -.0005)
                             | (cup_points[:, 2] > spec.get("body_top_m", spec["height_m"])+.0005))
            handle_only = bool(np.all(distance <= spec["handle_tube_radius_m"]+.001)
                               and outside_body.all())
        center_world = p+r.apply(center)
        seat_in_hole = r.inv().apply(seat-p)-center
        seated = ((seat_in_hole[0]/inner_rx)**2+(seat_in_hole[2]/inner_rz)**2 <= 1.08**2
                  and abs(seat_in_hole[1]) < .065)
        state.update(hook_through_handle=bool(self.threaded and seated),
                     handle_above_hook=bool(center_world[2] > seat[2]+.005),
                     lowering_m=0. if self.threading_height is None else self.threading_height-float(center_world[2]),
                     hook_up_force_n=upward if handle_only else 0.,
                     other_support_force_n=state["table_force_n"]+(0. if handle_only else upward),
                     hook_contact_on_handle=handle_only,
                     release_clearance_m=self._robot_clearance(p, r))

    def _release_geometry(self):
        """Cache enabled gripper collider boxes in each actual rigid-body frame."""
        if hasattr(self, "_release_collision_bounds"):
            return self._release_collision_bounds
        from itertools import product
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = self.env.sim.stage
        robot = self.env.scene["robot"]
        root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        required = {"gelpad_left", "gelpad_right", "gelsight_mini_case_left", "gelsight_mini_case_right"}
        gripper = required | {"panda_hand", "panda_leftfinger", "panda_rightfinger"}
        transforms = UsdGeom.XformCache()
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
            ["default", "render", "proxy", "guide"], useExtentsHint=False)
        rows, covered = [], set()
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if (not prim.HasAPI(UsdPhysics.CollisionAPI)
                    or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False):
                continue
            body = prim
            while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
                body = body.GetParent()
            if not body or body.GetName() not in gripper:
                continue
            box = bounds.ComputeUntransformedBound(prim).ComputeAlignedRange()
            if box.IsEmpty():
                raise RuntimeError(f"Missing release-clearance collision bounds: {prim.GetPath()}")
            matrix = transforms.GetLocalToWorldTransform(prim)*transforms.GetLocalToWorldTransform(body).GetInverse()
            corners = np.asarray([matrix.Transform(Gf.Vec3d(*point)) for point in
                                  product(*zip(box.GetMin(), box.GetMax()))])
            if not np.isfinite(corners).all():
                raise RuntimeError(f"Nonfinite release-clearance collision bounds: {prim.GetPath()}")
            rows.append((robot.body_names.index(body.GetName()), corners))
            covered.add(body.GetName())
        if not required <= covered:
            raise RuntimeError(f"Missing release-clearance gripper colliders: {sorted(required-covered)}")
        self._release_collision_bounds = rows
        return rows

    def _robot_clearance(self, p, r):
        """Lower bound from all gripper collider boxes to the full mug bounds.

        Transform every enabled gel, housing, palm and finger collider using
        native body poses. Separate boxes avoid the full arm's huge AABB. The
        enclosing boxes fill concavities, so the result can reject a genuinely
        clear pose but cannot certify clearance by measuring pad centers only.
        """
        robot = self.env.scene["robot"]
        cached = getattr(self, "_cpu_cache", {})
        positions = cached.get("robot/body_positions")
        quaternions = cached.get("robot/body_quaternions")
        if positions is None:
            positions = robot.data.body_pos_w[0].detach().cpu().numpy()
            quaternions = robot.data.body_quat_w[0].detach().cpu().numpy()
        low, high = np.asarray(self.spec["view_bounds_local_m"])
        distances = []
        for index, corners in self._release_geometry():
            world = rotation(quaternions[index]).apply(corners)+positions[index]
            local = r.inv().apply(world-p)
            separation = np.maximum(np.maximum(low-local.max(axis=0), local.min(axis=0)-high), 0.)
            distances.append(float(np.linalg.norm(separation)))
        return min(distances)

    def _slip_state(self, state, p, r, dt):
        hand_p, hand_r = self.hand_pose()
        relative = hand_r.inv().apply(p-hand_p)
        object_relative_r = hand_r.inv()*r
        suspended = self.lowest_point(p, r) >= self.table_height+.03
        two_pads = min(state["left_gel_force_n"], state["right_gel_force_n"]) >= .05
        cached_hand = getattr(self, "_cpu_cache", {}).get("robot/hand")
        if cached_hand is not None:
            hand_speed = float(np.linalg.norm(cached_hand[7:10]))
            hand_angular_speed = float(np.linalg.norm(cached_hand[10:13]))
        else:
            robot = self.env.scene["robot"]
            hand_index = robot.body_names.index("panda_hand")
            hand_speed = float(robot.data.body_lin_vel_w[0, hand_index].norm())
            hand_angular_speed = float(robot.data.body_ang_vel_w[0, hand_index].norm())
        axis_angle = math.acos(float(np.clip(abs(r.apply([0., 0., 1.])[2]), 0., 1.)))
        quiet = (state["object_speed_m_s"] < .01 and state["angular_speed_rad_s"] < .1
                 and hand_speed < .005 and hand_angular_speed < .05
                 and axis_angle <= self.goal["max_axis_tilt_rad"])
        if self.slip_reference is None:
            self.arm_stable_s = self.arm_stable_s+dt if suspended and two_pads and quiet else 0.
            if self.arm_stable_s >= .2:
                self.slip_reference = (relative.copy(), object_relative_r, hand_p.copy(), hand_r)
        axial = axial_speed = hand_translation = hand_angle = relative_angle = axial_yaw = 0.
        relative_quaternion = [1., 0., 0., 0.]
        if self.slip_reference is not None:
            initial_p, initial_r, reference_hand_p, reference_hand_r = self.slip_reference
            axis = reference_hand_r.inv().apply([0., 0., -1.])
            axial = float(np.dot(relative-initial_p, axis))
            if dt > 0:
                axial_speed = (axial-getattr(self, "previous_axial", 0.))/dt
            self.previous_axial = axial
            hand_translation = float(np.linalg.norm(hand_p-reference_hand_p))
            hand_angle = float((reference_hand_r.inv()*hand_r).magnitude())
            relative_rotation = initial_r.inv()*object_relative_r
            relative_angle = float(relative_rotation.magnitude())
            q = relative_rotation.as_quat()
            relative_quaternion = [float(q[3]), *q[:3].tolist()]
            # Keep full rotation and axial twist as diagnostics. The task
            # constrains tilt without imposing an axial orientation target.
            axial_yaw = float((2*math.atan2(q[2], q[3])+math.pi) % (2*math.pi)-math.pi)
        for side in ("left", "right"):
            present = state[f"{side}_gel_force_n"] >= .05
            count, checked = 0, False
            if self.slip_reference is not None and not present:
                sensor = self.env.scene[f"gsmini_{side}_contact"]
                *_, counts, _ = sensor.contact_physx_view.get_contact_data(dt=self.env.cfg.sim.dt)
                counts = counts.detach().cpu().numpy().reshape(-1)
                if len(counts) != len(sensor.cfg.filter_prim_paths_expr) or (counts < 0).any():
                    raise RuntimeError("Slip native contact counts do not match target filters")
                count, checked = int(counts[0]), True
                reaction = max(np.linalg.norm(state[f"{side}_gel_{component}_force_w_n"])
                               for component in ("normal", "tangent"))
                # Native counts include near-contact buffer entries. Require
                # an actual target reaction above numerical noise as well.
                # The .05 N preload threshold remains required for arming/hold.
                present = count > 0 and reaction > 1e-6
            state.update({f"{side}_gel_contact_present": bool(present),
                          f"{side}_gel_low_force_native_contact_count": count,
                          f"{side}_gel_contact_count_checked": checked})
        state.update(slip_armed=self.slip_reference is not None, suspended=bool(suspended),
                     axial_slip_m=axial, axial_speed_m_s=axial_speed,
                     gripper_command=float(self._cpu_cache["robot/gripper_command"]),
                     gripper_aperture_m=float(self._cpu_cache["robot/gripper_aperture"]),
                     housing_contact_load_n=float(self._cpu_cache["robot/housing_contact_load"]),
                     hand_translation_m=hand_translation, hand_rotation_rad=hand_angle,
                     relative_rotation_rad=relative_angle,
                     relative_quaternion_wxyz=relative_quaternion, axial_yaw_rad=axial_yaw,
                     rod_axis_tilt_rad=axis_angle,
                     external_support_force_n=state["table_force_n"]+max(
                         0., state["robot_force_n"]-state["gel_total_force_n"]-state["housing_total_force_n"]),
                     housing_force_n=state["housing_total_force_n"])
        if not math.isfinite(state["housing_contact_load_n"]):
            raise RuntimeError("Nonfinite native housing contact load")

    def contact_points(self, name, *, loaded_only=False):
        """Read only populated native contact buffers, including cooked separation."""
        sensor = self.env.scene[name]
        force, points, _, distances, counts, starts = sensor.contact_physx_view.get_contact_data(
            dt=self.env.cfg.sim.dt)
        counts = counts.detach().cpu().numpy().reshape(-1)
        starts = starts.detach().cpu().numpy().reshape(-1)
        indices = np.concatenate([np.arange(start, start+count) for start, count in zip(starts, counts)
                                  if count > 0]) if counts.sum() else np.empty(0, dtype=int)
        if not len(indices):
            return np.empty((0, 3)), np.empty(0)
        points = points.detach().cpu().numpy()[indices]
        distances = distances.detach().cpu().numpy().reshape(-1)[indices]
        if not np.isfinite(points).all() or not np.isfinite(distances).all():
            raise RuntimeError(f"Nonfinite cooked contact geometry: {name}")
        if loaded_only:
            loads = force.detach().cpu().numpy().reshape(-1)[indices]
            if not np.isfinite(loads).all():
                raise RuntimeError(f"Nonfinite cooked contact load: {name}")
            # PhysX also retains speculative patches with zero reaction.
            points, distances = points[loads > 0.], distances[loads > 0.]
        return points, distances

    def penetration_report(self):
        names = ([f"audit_{name}_contact" for name in ("small", "medium", "large")]
                 if self.kind == "gears" else ["support_contact", "object_robot_contact"])
        if self.kind == "hang":
            names.append("hook_contact")
        rows = []
        for name in names:
            _, distances = self.contact_points(name)
            rows.append({"sensor": name, "contact_count": len(distances),
                         "minimum_separation_m": float(distances.min()) if len(distances) else None})
        passed = all(row["minimum_separation_m"] is None or row["minimum_separation_m"] >= -.001
                     for row in rows)
        if self.kind != "gears":
            normal, tangent = self.force("support_contact")
            support_up = float((normal+tangent).sum(axis=0)[2])
            weight = self.spec["mass_kg"]*9.81
            # An empty or incorrectly resolved table filter cannot certify a
            # resting free body, even when its native pose looks supported.
            passed &= .8*weight <= support_up <= 1.2*weight
        return {"passed": passed, "status": "measured", "tolerance_m": .001,
                "scope": "Populated native cooked contact pairs: task/table, task/robot, task/hook or gear pairs; joint-represented bearings excluded.",
                "pairs": rows}

    def initial_state(self):
        """Snapshot the resolved native reset for diagnostic replay."""
        return native_initial_state(self.env)

    def restore(self, state):
        """Reset-only restoration; never used by an expert during an episode."""
        restore_native_initial_state(self.env, state)

    def close(self):
        self.env.scene.update = self.original_update
