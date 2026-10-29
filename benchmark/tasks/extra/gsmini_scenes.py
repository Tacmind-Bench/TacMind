"""Extra scene contact experts and collection with the shared recorder.

The shared expert collector dispatches these strategies. A recording is
published only after the physical task succeeds; failed attempts stay separate.
"""

from __future__ import annotations

from contextlib import ExitStack
import math
from pathlib import Path
from types import SimpleNamespace

PHASES = ("settle", "approach", "grasp", "lift", "transport", "insert",
          "actuate", "release", "retreat", "verify")
_FAST_MOTION_SCENES = {"coffee", "stack_blocks", "classify_objects", "bulb"}

def _rotation(quaternion):
    """Convert live WXYZ tensors to SciPy's XYZW convention."""
    from scipy.spatial.transform import Rotation
    q = quaternion.detach().cpu().numpy()
    return Rotation.from_quat([*q[1:], q[0]])


def _nearest_grasp_quaternion(quaternion, reference):
    """Parallel jaws admit a half-turn about approach; avoid a wrist-limit branch."""
    from scipy.spatial.transform import Rotation
    goal = Rotation.from_quat([*quaternion[1:], quaternion[0]])
    current = Rotation.from_quat([*reference[1:], reference[0]])
    choices = (goal, goal * Rotation.from_euler("z", math.pi))
    chosen = min(choices, key=lambda r: (current.inv()*r).magnitude())
    x, y, z, w = chosen.as_quat()
    return [w, x, y, z]


class RobotMotion:
    """Existing bounded Panda IK and collision planning; commands only, no state writes."""

    def __init__(self, env, app, on_step=None):
        from benchmark.backends.gsmini.motion import _make_pose_controller
        self.env, self.app = env, app
        self.on_step, self.grasp_env = on_step, None
        env.step_observer = self._observe
        self.controller = _make_pose_controller(env)
        self.phase("settle")

    def _observe(self):
        if self.on_step is not None:
            self.on_step()
        if self.grasp_env is not None:
            from benchmark.backends.gsmini.motion import _check_grasp_motion
            self.grasp_env.common_step_counter = self.env.common_step_counter
            _check_grasp_motion(self.grasp_env)

    def phase(self, name):
        self.env.expert_phase = PHASES.index(name)

    def wait(self, predicate, label, *, seconds=3.):
        for _ in range(math.ceil(seconds / self.env.step_dt)):
            if bool(predicate()):
                return
            self.step()
        if not bool(predicate()):
            raise RuntimeError(f"Extra scene stage unverified: {label}")

    def step(self):
        if not self.app.is_running() or self.env.sim.is_stopped():
            raise RuntimeError("Simulation closed during Extra scene expert")
        _, _, _, truncated, info = self.env.step(self.env.action)
        if truncated.any():
            raise RuntimeError("Extra scene expert timed out")
        return info["task_success"]

    def hold(self, seconds):
        for _ in range(math.ceil(seconds / self.env.step_dt)):
            self.step()

    def tcp(self):
        return self.env.scene["ee_frame"].data.target_pos_w[0, 0].detach().cpu().numpy().copy()

    def move(self, position, quaternion=(0., 1., 0., 0.), *, seconds=3., tolerance=.004,
             angular_tolerance=math.radians(3.), until=None):
        import torch
        from benchmark.backends.gsmini.workspace import grasp_poses
        from benchmark.backends.gsmini.motion import _hand_pose_joint_target
        env = self.env
        point = torch.as_tensor(position, dtype=torch.float32, device=env.device).reshape(1, 3)
        goal = grasp_poses(env, point, quaternion=quaternion)
        rotation = torch.as_tensor(quaternion, dtype=torch.float32, device=env.device)
        rotation = rotation / rotation.norm()
        arm = env.action_manager.get_term("arm_action")
        kind = env.task["source"]["scene"]
        reached = 0
        for _ in range(math.ceil(seconds/env.step_dt)):
            if until is not None and bool(until()):
                return
            holding_pod = (kind == "coffee" and self.grasp_env is not None
                           and self.grasp_env.scene["object"] is env.upstream_scene.pod)
            targets = _hand_pose_joint_target(env, self.controller, goal, center_joints=True,
                max_joint_step=.08 if not holding_pod and kind in _FAST_MOTION_SCENES else .04)
            env.action[:, :arm.action_dim] = arm.position_actions(targets)
            self.step()
            error = float((env.scene["ee_frame"].data.target_pos_w[:, 0] - point).norm())
            actual = env.scene["ee_frame"].data.target_quat_w[0, 0]
            cosine = (actual / actual.norm() * rotation).sum().abs().clamp(0., 1.)
            angle = 2 * float(torch.acos(cosine))
            reached = reached + 1 if error < tolerance and angle < angular_tolerance else 0
            empty_coffee_motion = (kind == "coffee" and self.grasp_env is None
                                   and env.expert_phase in (PHASES.index("approach"), PHASES.index("retreat")))
            if reached >= (1 if empty_coffee_motion else 3) and until is None:
                return
        if until is not None and bool(until()):
            return
        raise RuntimeError(f"Extra scene robot did not reach TCP target {list(position)}: "
                           f"error={error:.5f} m, angle={math.degrees(angle):.2f} deg")


    def contact_forces(self, asset, *, kind="gel"):
        """Read both sensors against this asset, including articulated links."""
        import torch
        path = asset.cfg.prim_path
        forces = []
        for side in ("left", "right"):
            sensor = self.env.scene[f"gsmini_{side}_{'case_' if kind == 'case' else ''}contact"]
            indices = [i for i, p in enumerate(sensor.cfg.filter_prim_paths_expr)
                       if p == path or p.startswith(path + "/")]
            if not indices:
                raise RuntimeError(f"Extra scene asset missing from contact filter: {path}")
            forces.append(sensor.data.force_matrix_w[0, :, indices, :].sum(dim=(0, 1)))
        result = torch.stack(forces)
        if not torch.isfinite(result).all():
            raise RuntimeError(f"Non-finite Extra scene grasp contact: {path}")
        return result

    def pinch_point(self, point, quaternion):
        """Align the desired grasp with actual GSmini gel centres, as in Recog."""
        from scipy.spatial.transform import Rotation
        from benchmark.backends.gsmini.sensors import pad_centers_world
        left, right = pad_centers_world(self.env)
        midpoint = ((left + right) / 2)[0].detach().cpu().numpy()
        actual = _rotation(self.env.scene["ee_frame"].data.target_quat_w[0, 0])
        desired = Rotation.from_quat([*quaternion[1:], quaternion[0]])
        target = point - desired.apply(actual.inv().apply(midpoint - self.tcp()))
        if not hasattr(self, "gripper_corners"):
            self.gripper_corners = _gripper_collision_corners(self.env.cfg)
        # Thin keys and pens lie below the gel centre's table clearance. Pinch
        # their upper edge with the lower gel, keeping the actual housing clear.
        floor = self.env.upstream_scene.cfg.surface_z + float(self.env.scene.env_origins[0, 2])
        target[2] = max(target[2], floor - desired.apply(self.gripper_corners)[:, 2].min() + .001)
        return target

    def close_on(self, asset, *, seconds=3., load_bearing=True, contact_kind="gel"):
        """Progressive physical closure, stopping only after bilateral gel contact."""
        self.phase("grasp")
        # Weight is a preload target, not a friction/load test: the housing can
        # share contact with the gel. Confirm bilateral contact at full closure,
        # then let the monitored physical lift test the load path.
        mass = float(asset.root_physx_view.get_masses().sum()) if load_bearing else 0.
        minimum_force = max(.1, mass*9.81)
        kind = self.env.task["source"]["scene"]
        closed_hold = .15 if kind == "nut_thread" or (
            kind == "bulb" and self.env.task_result["stages"]["engaged"] >= 1.) else .5
        stable, closed_contact = 0, 0
        for _ in range(math.ceil(seconds / self.env.step_dt)):
            if stable == 0 or contact_kind == "case":
                step = .1
                if kind in {"coffee", "bulb"}:
                    force = self.contact_forces(asset, kind=contact_kind).norm(dim=-1)
                    if bool((force < .005).all()):
                        step = .2
                self.env.action[:, -1] = (self.env.action[:, -1] - step).clamp(min=-1.)
            self.step()
            forces = (self.contact_forces(asset) if contact_kind == "gel" else
                      self.contact_forces(asset, kind=contact_kind)).norm(dim=-1)
            stable = stable + 1 if bool((forces >= minimum_force).all()) else 0
            at_limit = bool((self.env.action[:, -1] <= -1.).all())
            threshold = minimum_force if contact_kind == "case" else .05
            closed_contact = closed_contact + 1 if at_limit and bool((forces >= threshold).all()) else 0
            if ((contact_kind == "gel" and stable >= 3)
                    or closed_contact >= math.ceil(closed_hold / self.env.step_dt)):
                if load_bearing:
                    from benchmark.backends.gsmini.motion import _tcp_object_pose
                    self.grasp_env = SimpleNamespace(cfg=self.env.cfg, step_dt=self.env.step_dt, scene={
                        "object": asset, "robot": self.env.scene["robot"], "ee_frame": self.env.scene["ee_frame"]})
                    self.grasp_env.grasp_reference = _tcp_object_pose(self.grasp_env)
                return
        raise RuntimeError(f"Extra scene grasp lacks sustained bilateral {contact_kind} contact: {asset.cfg.prim_path}; "
                           f"forces={forces.tolist()}, preload_target={minimum_force:.3f} N")


    def planned(self, position, quaternion=(0., 1., 0., 0.), *, exclude=(), payload=None, velocity_scale=1.):
        import torch
        from benchmark.backends.gsmini.workspace import read_collision_world, grasp_poses
        from benchmark.backends.gsmini.motion import _create_motion_planner, _plan_pose, _attach_payload_spheres, _planner_joint_state
        from benchmark.backends.gsmini.sensors import planner_robot
        env = self.env
        payload_names = {name for name, asset in env.scene.rigid_objects.items() if asset is payload}
        paths = tuple(getattr(env.cfg.scene, name).prim_path
                      for name in env.cfg.extra_scene_asset_names
                      if name not in {*exclude, *payload_names, "ground", "light"})
        world = read_collision_world(env, obstacle_paths=paths)
        def robot_config(value):
            config = planner_robot(value, attached_sphere_count=16 if payload is not None else 4)
            robot = value.scene["robot"]
            locks = config["robot_cfg"]["kinematics"]["lock_joints"]
            for name in locks:
                locks[name] = float(robot.data.joint_pos[0, robot.joint_names.index(name)])
            return config
        from unittest.mock import patch
        from curobo._src.geom.data.data_mesh import MeshData, WarpMeshCache
        def load_mesh(data, mesh):
            import numpy as np
            import warp as wp
            vertices, faces = mesh.get_mesh_data()
            v = wp.array(vertices, dtype=wp.vec3, device=data._wp_device)
            f = wp.array(np.ravel(faces), dtype=wp.int32, device=data._wp_device)
            # Warp 1.10.1 SAH crashes/hangs on the original stapler scan.
            # Median indexes the identical vertices and triangles.
            native = wp.Mesh(points=v, indices=f, bvh_constructor="median")
            return WarpMeshCache(mesh.name, native.id, v, f, native)
        with patch.object(MeshData, "_load_mesh_to_warp", load_mesh):
            kind = env.task["source"]["scene"]
            holding_pod = kind == "coffee" and payload is env.upstream_scene.pod
            planner_options = {"trajectory_knots": 6} if kind in _FAST_MOTION_SCENES and not holding_pod else {}
            planner = _create_motion_planner(env, 0, world, robot_config=robot_config, velocity_scale=velocity_scale,
                **planner_options)
        attached = False
        try:
            if payload is not None:
                from curobo.types import Pose
                from curobo.sphere_fit import SphereFitType
                geometry = read_collision_world(env, obstacle_paths=(payload.cfg.prim_path,))
                attached = True
                manager = planner.attachment_manager
                _attach_payload_spheres(manager, manager.attach,
                    _planner_joint_state(env.scene["robot"], planner), list(geometry.objects),
                    num_spheres=16, sphere_fit_type=SphereFitType.VOXEL,
                    world_objects_pose_offset=Pose.from_list([0,0,0,1,0,0,0], planner.device_cfg))
            goal = grasp_poses(env, torch.as_tensor(position, dtype=torch.float32,
                device=env.device).reshape(1, 3), quaternion=quaternion)
            _plan_pose(env, self.app, planner, env.action, goal, "extra_scene_transport",
                settle_seconds=.05 if kind == "coffee" and payload is None else .1)
        finally:
            if attached:
                planner.attachment_manager.detach()
            planner.destroy()


    def grip(self, command, *, seconds=.7):
        if command > 0:
            self.grasp_env = None
        self.env.action[:, -1] = command
        if command == 1.:
            # Match tabletop experts' two measured open samples, using Extra scene's
            # step so an earned goal does not abort the remaining release.
            robot = self.env.scene["robot"]
            ids, _ = robot.find_joints(self.env.cfg.actions.gripper_action.joint_names)
            target = self.env.action_manager.get_term("gripper_action")._open_command
            opened = 0
            for _ in range(math.ceil(seconds / self.env.step_dt)):
                self.step()
                error = float((robot.data.joint_pos[:, ids] - target).abs().max())
                if not math.isfinite(error):
                    raise RuntimeError("Nonfinite gripper state during opening")
                opened = opened + 1 if error <= .001 else 0
                if opened >= 2:
                    return
            raise RuntimeError("Extra scene gripper did not fully open before retreat")
        else:
            self.hold(seconds)

    def take(self, asset, *, point=None, quaternion=(0., 1., 0., 0.), yaw_symmetry=2,
             planned_lift=False, contact_kind="gel"):
        import numpy as np
        from scipy.spatial.transform import Rotation
        from benchmark.backends.gsmini.environment import PlanningFailed
        if yaw_symmetry not in (2, 4):
            raise ValueError("Grasp yaw symmetry must be 2 (parallel jaws) or 4 (square faces)")
        quaternion = _nearest_grasp_quaternion(quaternion,
            self.env.scene["ee_frame"].data.target_quat_w[0, 0].cpu().tolist())
        rotation = Rotation.from_quat([*quaternion[1:], quaternion[0]])
        candidates = [quaternion]
        for angle in ((180., 90., 270.) if yaw_symmetry == 4 else (180.,)):
            x, y, z, w = (rotation * Rotation.from_euler("z", angle, degrees=True)).as_quat()
            candidates.append([w, x, y, z])
        point = asset.data.root_pos_w[0].detach().cpu().numpy().copy() if point is None else np.array(point)
        before = float(asset.data.root_pos_w[0, 2])
        kind = self.env.task["source"]["scene"]
        self.phase("approach")
        self.grip(1.)
        # Let the global planner choose a reachable elbow branch near contact;
        # a long local descent can hit joint 4's limit beside the robot base.
        for index, quaternion in enumerate(candidates):
            aligned = self.pinch_point(point, quaternion)
            try:
                self.planned(aligned + [0., 0., .02 if kind == "coffee" else .03], quaternion)
            except PlanningFailed:
                # Only planning rejection permits another equivalent grasp.
                # Execution/contact failures must retain their failure evidence.
                if index == len(candidates) - 1:
                    raise
            else:
                point = aligned
                break
        self.move(point, quaternion, seconds=4., tolerance=.001, angular_tolerance=math.radians(1.))
        if contact_kind == "gel":
            self.close_on(asset)
        else:
            self.close_on(asset, contact_kind=contact_kind)
        self.phase("lift")
        lift = point + [0., 0., .06 if kind == "stack_blocks" else .09]
        if planned_lift:
            self.planned(lift, quaternion, payload=asset)
        else:
            self.move(lift, quaternion, seconds=4., tolerance=.008)
        if float(asset.data.root_pos_w[0, 2]) - before < .025:
            raise RuntimeError("Extra scene grasp did not physically lift the object")
        self.hold(.2 if kind == "coffee" else .4)
        forces = (self.contact_forces(asset) if contact_kind == "gel" else
                  self.contact_forces(asset, kind=contact_kind)).norm(dim=-1)
        threshold = .05 if contact_kind == "gel" else max(.1, float(asset.root_physx_view.get_masses().sum())*9.81)
        if not bool((forces >= threshold).all()):
            raise RuntimeError("Extra scene object slipped from the physical grasp")


    def body_target(self, asset, position, quaternion):
        import torch
        from curobo.types import Pose
        env = self.env
        data = asset.data
        frame = env.scene["ee_frame"].data
        body = Pose(data.root_pos_w.clone(), data.root_quat_w.clone())
        tcp = Pose(frame.target_pos_w[:, 0].clone(), frame.target_quat_w[:, 0].clone())
        relative = body.inverse().multiply(tcp)
        goal = Pose(torch.tensor([list(position)], dtype=torch.float32, device=env.device),
                    torch.tensor([list(quaternion)], dtype=torch.float32, device=env.device)).multiply(relative)
        return goal.position[0].cpu().numpy(), goal.quaternion[0].cpu().tolist()

    def place(self, asset, position, quaternion=(1., 0., 0., 0.), *, release=True,
              approach_height=.12, release_check=None):
        self.phase("transport")
        point, rotation = self.body_target(asset, position, quaternion)
        self.planned(point + [0., 0., approach_height], rotation, payload=asset)
        self.phase("insert")
        kind = self.env.task["source"]["scene"]
        precision_placement = kind == "bulb" or (kind == "coffee" and asset is self.env.upstream_scene.pod)
        self.move(point, rotation, seconds=5., tolerance=.001 if precision_placement else .004)
        if release:
            self.phase("release")
            self.grip(1.)
            self.phase("retreat")
            from curobo.types import Pose
            from benchmark.tasks.success import PoseStabilityWindow
            # Confirm for 0.5 s while retreating. A task-owned check verifies its
            # native goal; other placements use the same pose window as tabletop tasks.
            # Late motion or a native-goal miss resets the confirmation.
            def pose():
                return Pose(asset.data.root_pos_w, asset.data.root_quat_w)
            window = PoseStabilityWindow(pose(), step_dt=self.env.step_dt)
            previous = self.on_step
            verified = 0
            def observe_release():
                nonlocal verified
                if previous is not None:
                    previous()
                window.observe(pose())
                if release_check is not None:
                    verified = verified+1 if bool(release_check()) else 0
            self.on_step = observe_release
            try:
                self.move(point + [0., 0., .12], rotation)
                if release_check is None:
                    self.wait(lambda: window.ready, "released object stability", seconds=.7)
                else:
                    self.wait(lambda: verified >= window.steps, "released object native goal")
            finally:
                self.on_step = previous


def _blocks(env, motion):
    from scipy.spatial.transform import Rotation
    scene, cfg = env.upstream_scene, env.upstream_scene.cfg
    stacking = env.task["source"]["scene"] == "stack_blocks"
    origin = scene.env_origins[0].cpu().numpy()
    for index, (name, asset) in enumerate(scene.blocks.items()):
        point, hand_q = _site_grasp((name, asset, (-cfg.block_size/2, 0., 0.),
                                    (cfg.block_size/2, 0., 0.), None))
        # The close sorting arc can drive local lift IK into a wrist limit.
        # Plan that lift with the held block's geometry on the robot.
        motion.take(asset, point=point, quaternion=hand_q, yaw_symmetry=4,
                    planned_lift=not stacking)
        if stacking:
            target = [*cfg.pad_pos, cfg.surface_z + cfg.block_size * (index + .5) + .001]
        else:
            category = dict(cfg.manifest)[name]
            target = [*scene._zones[0, category].cpu().tolist(), cfg.surface_z + cfg.block_size/2 + .001]
        # These graders require upright faces, not a prescribed yaw. Preserve
        # the measured yaw instead of demanding an unnecessary wrist rewind.
        yaw = _rotation(asset.data.root_quat_w[0]).as_euler("xyz")[2]
        q = Rotation.from_euler("z", yaw).as_quat()
        motion.place(asset, origin + target, quaternion=[q[3], *q[:3]], approach_height=.03 if stacking else .12)


def _site_grasp(site):
    """Use upstream grip bands with the jaw across, rather than along, a part."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    _, asset, lo, hi, _ = site
    frame = _rotation(asset.data.root_quat_w[0])
    if isinstance(lo, str) and lo == "circle":
        # Pinch one wall of the mug/pod, not the full outside diameter.
        radius, height = hi
        point = asset.data.root_pos_w[0].cpu().numpy() + frame.apply([0., -radius, height])
        jaw = frame.apply([0., 1., 0.])
        jaw[2] = 0.
        jaw /= np.linalg.norm(jaw)
        down = np.array([0., 0., -1.])
        q = Rotation.from_matrix(np.stack((np.cross(jaw, down), jaw, down), axis=1)).as_quat()
        return point, [q[3], *q[:3]]
    a,b = np.asarray(lo),np.asarray(hi)
    axis = frame.apply(b-a)
    down = np.array([0.,0.,-1.])
    jaw = np.cross(axis,down)
    if np.linalg.norm(jaw) < 1e-6:
        raise RuntimeError("The grip band cannot be approached with a top-down parallel jaw")
    jaw /= np.linalg.norm(jaw)
    hand = Rotation.from_matrix(np.stack((np.cross(jaw,down),jaw,down),axis=1)).as_quat()
    return asset.data.root_pos_w[0].cpu().numpy()+frame.apply((a+b)/2), [hand[3],*hand[:3]]


def _thread_sweep(motion, axis, angle, pose, measure, *, until=None):
    import numpy as np
    import torch
    from benchmark.backends.gsmini.workspace import grasp_poses
    from benchmark.backends.gsmini.motion import _hand_pose_joint_target
    env = motion.env
    arm = env.action_manager.get_term('arm_action')
    bulb = env.task["source"]["scene"] == "bulb"
    speed = 3.5 if bulb and until is not None else 6.
    controller = motion.controller
    if not bulb:
        # Distribute the nut turn across the arm instead of saturating its wrist.
        weights = torch.tensor([1., 1., 1., 1., 1., 1., .4], device=env.device)

        def coordinated_compute(position, quaternion, jacobian, joints):
            proposed = motion.controller.compute(position, quaternion, jacobian*weights, joints)
            return joints+(proposed-joints)*weights

        controller = SimpleNamespace(set_command=motion.controller.set_command, compute=coordinated_compute)
    previous, actual, pending, stable = measure(), 0., 0., 0
    sign = math.copysign(1., angle)
    for _ in range(math.ceil((abs(angle)/.65+1.)/env.step_dt)):
        if until is not None and until():
            return
        current = measure()
        actual += sign*float(((current*previous.inv()).as_rotvec()@axis))
        previous = current
        pending = min(abs(angle), pending+speed*env.step_dt, max(0., actual)+(1.2 if bulb and until is None else .5))
        position, desired = pose(sign*pending)
        quaternion = desired.as_quat()[[3, 0, 1, 2]].tolist()
        goal = grasp_poses(env, torch.as_tensor(position, dtype=torch.float32,
            device=env.device).reshape(1,3), quaternion=quaternion)
        targets = _hand_pose_joint_target(env, controller, goal,
            center_joints=True, max_joint_step=.35 if bulb else .30)
        env.action[:, :arm.action_dim] = arm.position_actions(targets)
        motion.step()
        stable = stable+1 if (pending >= abs(angle) and
            (_rotation(env.scene["ee_frame"].data.target_quat_w[0,0]).inv()*desired).magnitude() < .04 and
            np.linalg.norm(motion.tcp()-position) < .004) else 0
        if stable >= 2:
            return
    raise RuntimeError(f'Threading wrist turn did not converge: desired={angle}, actual={actual}')


def _continuous_thread(env, motion, part, fixture, index=0):
    import numpy as np
    from scipy.spatial.transform import Rotation
    rotation = lambda asset: _rotation(asset.data.root_quat_w[0])
    hand = lambda: _rotation(env.scene['ee_frame'].data.target_quat_w[0,0])
    bulb = env.task["source"]["scene"] == "bulb"
    fixture_r = rotation(fixture)
    axis = fixture_r.apply([0.,0.,1.])
    # Retain the measured grasp band through each supported regrasp.
    from benchmark.backends.gsmini.sensors import pad_centers_world
    left, right = pad_centers_world(env)
    midpoint = ((left+right)/2)[0].cpu().numpy()
    grasp_local = rotation(part).inv().apply(midpoint-part.data.root_pos_w[0].cpu().numpy())
    begin = hand()
    q = begin.as_quat()[[3,0,1,2]].tolist()
    for cycle in range(20):
        if bool(env.upstream_scene.seated()[0,index]):
            break
        if cycle:
            point = part.data.root_pos_w[0].cpu().numpy()+rotation(part).apply(grasp_local)
            motion.move(motion.pinch_point(point, q), q, seconds=3., tolerance=.001)
            if not bulb:
                env.action[:, -1] = -1.
            motion.close_on(part)
        motion.phase('actuate')
        start_hand = hand()
        start_part = rotation(part)
        relative_r = start_part.inv()*start_hand
        relative_p = start_part.inv().apply(motion.tcp()-part.data.root_pos_w[0].cpu().numpy())
        centre = fixture.data.root_pos_w[0].cpu().numpy()
        offset = fixture_r.inv().apply(part.data.root_pos_w[0].cpu().numpy()-centre)
        offset[:2] = 0.
        origin = centre+fixture_r.apply(offset)
        yaw = (fixture_r.inv()*start_part).as_euler('xyz')[2]
        level = fixture_r*Rotation.from_euler('z',yaw)
        def pose(angle):
            r = Rotation.from_rotvec(axis*angle)*level
            p = origin+axis*(.002/(2*math.pi))*angle
            return p+r.apply(relative_p), r*relative_r
        _thread_sweep(motion, axis, -math.pi, pose, lambda: rotation(part),
            until=lambda: bool(env.upstream_scene.seated()[0,index]))
        if bool(env.upstream_scene.seated()[0,index]):
            break
        if bulb:
            motion.grip(1.)
        else:
            # The real fixture supports the nut while the open wrist rewinds.
            motion.grasp_env = None
            env.action[:, -1] = -.15
            robot = env.scene['robot']
            ids, _ = robot.find_joints(env.cfg.actions.gripper_action.joint_names)
            motion.wait(lambda: float(robot.data.joint_pos[0,ids].min()) >= .016,
                        'fixture-supported nut release', seconds=1.)
        tcp, start = motion.tcp(), hand()
        total = float(((begin*start.inv()).as_rotvec()@axis))
        # The actual half turn can be just over pi. Rewind in the direction
        # opposite to the loaded turn rather than taking quaternion shortest path.
        if total < 0.:
            total += 2*math.pi
        _thread_sweep(motion, axis, total,
              lambda a: (tcp, Rotation.from_rotvec(axis*a)*start), lambda: hand())
        print('[THREAD]', cycle, env.task_result, flush=True)
    if not bool(env.upstream_scene.seated()[0,index]):
        raise RuntimeError('Continuous wrist cycles did not physically seat the part')
    motion.grip(1.)
    if bulb:
        motion.move(motion.tcp()+[0.,0.,.1])
    else:
        motion.move(motion.tcp()+[0.,0.,.1],
                    env.scene['ee_frame'].data.target_quat_w[0,0].cpu().tolist())
    motion.hold(.6)


def _thread(env, motion):
    """Thread through bounded wrist turns and physical release/regrasp cycles."""
    scene, cfg = env.upstream_scene, env.upstream_scene.cfg
    bulb = env.task["source"]["scene"] == "bulb"
    parts = scene.bulbs if bulb else scene.nuts
    fixtures = scene.sockets if bulb else scene.bolts
    for index, (part, fixture) in enumerate(zip(parts, fixtures)):
        if bulb:
            # Both pads must pinch glass; an axial grasp bears on the slick cap.
            point, hand_q = _site_grasp(("bulb", part, (0., 0., .03), (0., 0., .05), None))
        else:
            frame = _rotation(part.data.root_quat_w[0])
            point = part.data.root_pos_w[0].cpu().numpy() + frame.apply([0., 0., .015])
            hand_q = [0., math.cos(math.pi/4), math.sin(math.pi/4), 0.]
        motion.take(part, point=point, quaternion=hand_q)
        centre = fixture.data.root_pos_w[0].cpu().numpy().copy()
        start = cfg.socket_opening_z if bulb else cfg.bolt_height
        motion.place(part, centre + [0.,0.,start], release=False, approach_height=.06 if bulb else .12)
        if bulb:
            # Rest on the real socket, then grasp the glass from above so the
            # wrist can turn about the socket axis without an eccentric load.
            current_q = env.scene["ee_frame"].data.target_quat_w[0, 0].cpu().tolist()
            motion.grip(1.)
            motion.move(motion.tcp() + [0.,0.,.05], current_q, seconds=4.)
            point = part.data.root_pos_w[0].cpu().numpy() + _rotation(
                part.data.root_quat_w[0]).apply([0., 0., .049])
            hand_q = [0., math.cos(math.pi/4), math.sin(math.pi/4), 0.]
            point = motion.pinch_point(point, hand_q)
            motion.planned(point + [0., 0., .05], hand_q)
            motion.move(point, hand_q, seconds=4., tolerance=.001)
            motion.close_on(part)
        _continuous_thread(env, motion, part, fixture, index)


def _gripper_collision_corners(cfg):
    """Read the hand, fingers and GSmini support clearance in the TCP frame."""
    import itertools
    import numpy as np
    from scipy.spatial.transform import Rotation
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.Open(cfg.scene.robot.spawn.usd_path)
    hand = stage.GetPrimAtPath("/panda/panda_hand")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"], False, True)
    corners = []
    names = ["panda_hand", "panda_leftfinger", "panda_rightfinger"]
    names += [name for side in ("left", "right")
              for name in (f"gelpad_{side}", f"gelsight_mini_case_{side}")]
    for name in names:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(f"/panda/{name}"), Usd.TraverseInstanceProxies()):
            if (prim.HasAPI(UsdPhysics.CollisionAPI)
                    and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False):
                box = cache.ComputeRelativeBound(prim, hand).ComputeAlignedRange()
                if not box.IsEmpty():
                    corners.extend(itertools.product(*zip(box.GetMin(), box.GetMax())))
    if not corners:
        raise RuntimeError("No GSmini collision geometry for table clearance")
    offset = cfg.scene.ee_frame.target_frames[0].offset
    q = offset.rot
    return Rotation.from_quat([*q[1:], q[0]]).inv().apply(np.asarray(corners)-offset.pos)


def _move_body(motion, asset, position, rotation, *, planned=False, seconds=4., velocity_scale=1.):
    point, quaternion = motion.body_target(asset, position, rotation)
    if planned:
        motion.planned(point, quaternion, payload=asset, velocity_scale=velocity_scale)
    else:
        motion.move(point, quaternion, seconds=seconds, tolerance=.002)


def _local_bounds(asset):
    """Use the same authored collision geometry as the shared tabletop grasp helpers."""
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics
    source = Usd.Stage.Open(asset.cfg.spawn.usd_path)
    root = source.GetDefaultPrim()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_,
        UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide], False, True)
    bounds = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(
                prim).GetCollisionEnabledAttr().Get() is not False:
            box = cache.ComputeRelativeBound(prim, root).ComputeAlignedRange()
            if not box.IsEmpty():
                bounds.append((np.array(box.GetMin()), np.array(box.GetMax())))
    if not bounds:
        raise RuntimeError(f"No collision geometry for Extra scene grasp: {asset.cfg.prim_path}")
    scale = np.asarray((1., 1., 1.) if asset.cfg.spawn.scale is None else asset.cfg.spawn.scale)
    return np.min([b[0] for b in bounds], axis=0)*scale, np.max([b[1] for b in bounds], axis=0)*scale


def _geometry_grasp(asset):
    import numpy as np
    lo, hi = _local_bounds(asset)
    centre = (lo + hi)/2
    long_axis = int(np.argmax((hi-lo)[:2]))
    a, b = centre.copy(), centre.copy()
    a[long_axis], b[long_axis] = lo[long_axis], hi[long_axis]
    return _site_grasp(("collision bounds", asset, a, b, None))


def _coffee(env, motion):
    import numpy as np
    from scipy.spatial.transform import Rotation
    scene, cfg = env.upstream_scene, env.upstream_scene.cfg
    origin = scene.env_origins[0].cpu().numpy()
    def machine(offset):
        return origin + np.asarray([cfg.cm_pos[0], cfg.cm_pos[1], cfg.surface_z]) + offset
    def take(asset):
        p, q = _site_grasp(next(s for s in scene.grasp_sites() if s[1] is asset))
        if asset is scene.cup:
            # Pinch the cup's inner wall against its handle's outer arc.
            # The 6 mm wall alone is thinner than the closed GSmini pad gap.
            body = _rotation(asset.data.root_quat_w[0])
            p = asset.data.root_pos_w[0].cpu().numpy()+body.apply([0., .062, .016])
            rotation = (body*Rotation.from_euler("x", 155., degrees=True)).as_quat()
            q = [rotation[3], *rotation[:3]]
        tcp = motion.pinch_point(p, q)
        hand = Rotation.from_quat([*q[1:], q[0]])
        # The pod's low grip band must clear the serving tray.
        floor = float(scene.env_origins[0, 2])+cfg.surface_z+cfg.tray_h
        minimum = floor-hand.apply(motion.gripper_corners)[:, 2].min()+.002
        p[2] += max(0., minimum-tcp[2])
        motion.take(asset, point=p, quaternion=q)
    def cover(opened):
        p, q = _site_grasp(scene.grasp_sites()[0])
        q = _nearest_grasp_quaternion(q, env.scene["ee_frame"].data.target_quat_w[0, 0].cpu().tolist())
        motion.phase("approach")
        motion.grip(1.)
        motion.planned(p+[0., 0., .05], q)
        p = motion.pinch_point(p, q)
        hand = Rotation.from_quat([*q[1:], q[0]])
        top = float(scene.cover.data.root_pos_w[0, 2])+cfg.cover_z[1]
        p[2] = max(p[2], top-hand.apply(motion.gripper_corners)[:, 2].min()+.002)
        motion.move(p, q)
        motion.close_on(scene.cover, load_bearing=False)
        motion.phase("actuate")
        travel = -cfg.cover_travel+.005 if opened else 0.
        motion.move(p + [0., travel-float(scene.cover_pos()[0]), 0.], q, seconds=5.,
                    until=scene.cover_open if opened else scene.cover_closed)
        motion.grip(1.)
        motion.move(motion.tcp()+[0., 0., .05], q)
        motion.wait(scene.cover_open if opened else scene.cover_closed, "coffee cover", seconds=1.)
    # Clear the actual appliance before the next collision-planned approach.
    motion.phase("approach")
    hand_q = env.scene["ee_frame"].data.target_quat_w[0, 0].cpu().tolist()
    target = motion.tcp()+[0., -.10, 0.]
    floor = cfg.surface_z+float(scene.env_origins[0, 2])
    hand=Rotation.from_quat([*hand_q[1:],hand_q[0]])
    clearance=-float(hand.apply(_gripper_collision_corners(env.cfg))[:,2].min())+.01
    target[2] = max(target[2], floor+cfg.cover_z[1]+clearance)
    motion.move(target, hand_q, seconds=5.)
    if not bool(scene.cover_open()[0]):
        cover(True)
    take(scene.pod)
    # Keep the full hand above the opening while gravity seats the pod.
    target = machine([*cfg.pocket_off, cfg.pocket_floor_z+cfg.pod_h/2+.002])
    tcp, q = motion.body_target(scene.pod, target, [1., 0., 0., 0.])
    hand = Rotation.from_quat([*q[1:], q[0]])
    top = origin[2]+cfg.surface_z+cfg.cover_z[1]
    target[2] += max(0., top-hand.apply(_gripper_collision_corners(env.cfg))[:, 2].min()+.02-tcp[2])
    motion.place(scene.pod, target, approach_height=0.)
    motion.wait(scene.pod_seated, "pod seated")
    cover(False)
    take(scene.cup)
    staged = machine([cfg.spout_off[0], cfg.spout_off[1]+cfg.cup_stage_y_off,
                      cfg.platform_top_z+cfg.cup_h/2+.002])
    # Clear the platform edge before crossing in front of the appliance.
    # Cup yaw is free in the native upright goal; keep the handle toward the front.
    cup_q = [math.cos(math.radians(-135.)/2), 0., 0., math.sin(math.radians(-135.)/2)]
    _move_body(motion, scene.cup, staged+[0., -.08, .06], cup_q, planned=True, velocity_scale=.9)
    _move_body(motion, scene.cup, staged+[0., -.08, .01], cup_q, seconds=4.)
    _move_body(motion, scene.cup, staged, cup_q, seconds=6.)
    motion.grip(1.)
    motion.move(motion.tcp()+[0., -.08, .08], env.scene["ee_frame"].data.target_quat_w[0, 0].cpu().tolist())
    motion.wait(scene.cup_under_spout, "cup under spout")
    motion.phase("actuate")
    button = machine(cfg.btn_off)+[0., 0., cfg.btn_size[2]/2]
    motion.grip(-1.,seconds=.1)
    motion.planned(button+[0., 0., .02])
    motion.move(button+[0., 0., -.003], seconds=4., until=lambda: scene.running()[0] or scene.filled()[0])
    motion.move(button+[0., 0., .04])
    motion.wait(scene.filled, "completed coffee brew", seconds=cfg.brew_steps*env.cfg.sim.dt+3.)


def _push_shapes(env, motion):
    """Slide with physical jaw contacts, releasing over each matching recess."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    scene = env.upstream_scene
    floor = scene.cfg.surface_z + scene.cfg.board_thickness + float(scene.env_origins[0, 2])
    previous = motion.on_step
    def check_board():
        if previous is not None:
            previous()
        for name in scene.blocks:
            if not bool(scene.not_lifted(name)[0]):
                raise RuntimeError(f"Push expert lifted {name} off the board")
    motion.on_step = check_board
    try:
        for name, asset in reversed(list(scene.blocks.items())):
            motion.phase("approach")
            motion.grip(1.)
            point, quaternion = _geometry_grasp(asset)
            if name == "x":
                # Pinch the real horizontal arm beyond its central crossing.
                point += _rotation(asset.data.root_quat_w[0]).apply(
                    [scene.cfg.piece_scale*.028, 0., 0.])
            quaternion = _nearest_grasp_quaternion(quaternion,
                env.scene["ee_frame"].data.target_quat_w[0, 0].cpu().tolist())
            hand = Rotation.from_quat([*quaternion[1:], quaternion[0]])
            point = motion.pinch_point(point, quaternion)
            point[2] = max(point[2], floor-hand.apply(_gripper_collision_corners(env.cfg))[:, 2].min()+.008)
            motion.planned(point+[0., 0., .06], quaternion)
            motion.move(point, quaternion, seconds=4., tolerance=.001)
            motion.close_on(asset)
            motion.phase("actuate")
            target = asset.data.root_pos_w[0].cpu().numpy().copy()
            target[2] = scene.cfg.slide_z+float(scene.env_origins[0, 2])
            pad = scene.pads[name]
            q = pad.data.root_quat_w[0].cpu().tolist()
            tcp, hand_q = motion.body_target(asset, target, q)
            motion.move(tcp, hand_q, seconds=4., tolerance=.004)
            goal = pad.data.root_pos_w[0].cpu().numpy().copy()
            for _ in range(40):
                current = asset.data.root_pos_w[0].cpu().numpy().copy()
                delta = goal-current
                delta[2] = 0.
                distance = np.linalg.norm(delta)
                if distance < .002:
                    break
                target = current + delta*min(1., .008/distance)
                target[2] = scene.cfg.slide_z+float(scene.env_origins[0, 2])
                tcp, hand_q = motion.body_target(asset, target, q)
                hand = Rotation.from_quat([*hand_q[1:], hand_q[0]])
                tcp[2] = max(tcp[2], floor-hand.apply(_gripper_collision_corners(env.cfg))[:, 2].min()+.008)
                motion.move(tcp, hand_q, seconds=2., tolerance=.004)
            robot = env.scene["robot"]
            ids, _ = robot.find_joints(env.cfg.actions.gripper_action.joint_names)
            gripper = env.action_manager.get_term("gripper_action")
            opening = (robot.data.joint_pos[0, ids].max()+.003).clamp(max=gripper._open_command.min())
            command = float((2*((opening-gripper._close_command)/(
                gripper._open_command-gripper._close_command)).max()-1).clamp(-1., 1.))
            motion.phase("release")
            motion.grasp_env = None
            motion.grip(command)
            motion.wait(lambda: scene.seated(name)[0], f"shape {name} seated")
            motion.phase("retreat")
            motion.move(motion.tcp()+[0., 0., .08], hand_q)
    finally:
        motion.on_step = previous


def wait_for_goal(motion, *, seconds=3.):
    """Allow released objects to settle and earn the existing stability window."""
    for _ in range(math.ceil(seconds / motion.env.step_dt)):
        if motion.env.task_result["success"]:
            return dict(motion.env.task_result)
        motion.step()
    if not motion.env.task_result["success"]:
        raise RuntimeError(f"Extra scene goal unverified: {motion.env.task_result}")
    return dict(motion.env.task_result)


EXPERTS = {"stack_blocks": _blocks, "classify_objects": _blocks,
           "bulb": _thread, "nut_thread": _thread,
           "coffee": _coffee, "push_shapes": _push_shapes}


def execute(env, app, *, on_step=None):
    """Run implemented physical experts; unknown mechanisms fail explicitly."""
    kind = env.task["source"]["scene"]
    if kind not in EXPERTS:
        raise NotImplementedError(f"Extra scene {kind}: physical GSmini expert is not implemented")
    previous_observer = env.step_observer
    try:
        motion = RobotMotion(env, app, on_step if on_step is not None else previous_observer)
        if kind!="coffee":
            motion.hold(.2 if kind in {"stack_blocks","bulb"} else 1.)
        EXPERTS[kind](env, motion)
        motion.phase("verify")
        return wait_for_goal(motion)
    finally:
        env.step_observer = previous_observer


def run(app, seed, output: Path, task_id, *, session=None):
    if session is not None:
        raise ValueError("Extra scene uses a fresh TacMind Franka + GSmini environment per episode")
    from benchmark.catalog import get_task, is_extra_scene
    from benchmark.expert.capabilities import supported_task_ids

    task = get_task("tac_extra", task_id)
    if not is_extra_scene(task):
        raise ValueError(f"Extra scene Expert does not handle mechanism task {task_id}")
    if task_id not in supported_task_ids("extra", "gsmini"):
        raise ValueError(f"Expert unavailable for Extra scene task {task_id}")
    from benchmark.tasks.extra.scenes import ExtraSceneEnvironment
    from benchmark.backends.gsmini.recording import _prepare_output_path, _start_episode, _finish_episode
    from benchmark.backends.gsmini.execution import configure_shutdown
    _prepare_output_path(output)
    with ExitStack() as resources:
        env = ExtraSceneEnvironment(task_id, seed=seed)
        resources.callback(env.close)
        configure_shutdown(env)
        env.expert_phase = 0
        stage_names = tuple(env.read_task_state()["stages"])
        def record_expert(episode):
            import torch
            from benchmark.backends.gsmini.recording import _record_tactile_observations
            _record_tactile_observations(env, episode)
            episode.add("extra_scene/phase", torch.tensor(env.expert_phase))
            state = env.read_task_state()
            episode.add("extra_scene/score", torch.tensor(state["score"]))
            episode.add("extra_scene/stages", torch.tensor([state["stages"][name] for name in stage_names]))
            kind = task["source"]["scene"]
            if kind in {"bulb", "nut_thread"}:
                scene = env.upstream_scene
                parts, fixtures = ((scene.bulbs, scene.sockets) if kind == "bulb"
                                   else (scene.nuts, scene.bolts))
                for label, assets in (("parts", parts), ("fixtures", fixtures)):
                    episode.add(f"extra_scene/thread_{label}_pose_w",
                                torch.stack([asset.data.root_state_w[0, :7] for asset in assets]))
            if task["source"]["scene"] == "push_shapes":
                episode.add("extra_scene/board_bottom_z", env.board_height_history.clone())
                episode.add("extra_scene/board_physics_steps", env.board_step_history.clone())
        env.record_tactile = record_expert
        staged = _start_episode(resources, env, output, "tac_extra", task_id, seed,
            {"task_definition": env.task, "expert_object_pose_writes_during_episode": 0,
             "expert_phases": PHASES, "expert_stage_names": stage_names,
             "expert_target_source": "live simulator geometry; physical robot actions only"}, task=env.task)
        try:
            result = execute(env, app)
        except Exception:
            import os
            import traceback
            from uuid import uuid4
            from benchmark.io import write_json
            from benchmark.backends.gsmini.recording import _flush_episode
            from benchmark.tasks.extra.scenes import state_snapshot
            failure = output.with_name(output.name + ".failed." + uuid4().hex[:8])
            diagnostic = {"task_id": task_id, "seed": seed, "error": traceback.format_exc(),
                "expert_phase": PHASES[env.expert_phase],
                "robot_joint_pos": env.scene["robot"].data.joint_pos[0].cpu().tolist(),
                "tcp_world": env.scene["ee_frame"].data.target_pos_w[0,0].cpu().tolist(),
                "physical_state": state_snapshot(env),
                "task_result": env.task_result}
            try:
                _flush_episode(env)
                env.handler.end_episode(success=False)
                env.handler.close()
                os.link(staged, failure)
                diagnostic["failed_recording"] = str(failure)
            except Exception:
                diagnostic["recording_error"] = traceback.format_exc()
            write_json(failure.with_suffix(failure.suffix + ".json"), diagnostic)
            print("EXTRA_SCENE_EXPERT_FAILURE", failure, flush=True)
            raise
        _finish_episode(env, output, staged)
        return result
