"""Shared Extra interaction paths, motion execution and native contact checks.

Extra experts share contact trajectories
and physics-rate history checks defined here.
"""

from __future__ import annotations

import math
import traceback

from benchmark.tasks.extra.contact import SAMPLE_FIELDS as CONTACT_STATE_FIELDS


def native_history_kind(kind):
    return kind in CONTACT_STATE_FIELDS or kind in {"stand", "press_slide", "surface_inspection", "ram"}


def _rotation(quaternion):
    from scipy.spatial.transform import Rotation
    w, x, y, z = quaternion
    return Rotation.from_quat([x, y, z, w])


def grasp_point(env):
    """Return the physical affordance centroid, using measured body transforms."""
    import numpy as np
    spec = env.task["objects"]["mechanism"]
    asset = env.scene["mechanism"]
    if spec.get("grasp_point_frame") == "root":
        position = asset.data.root_pos_w[0].cpu().numpy()
        rotation = _rotation(asset.data.root_quat_w[0].cpu().tolist())
        return position + rotation.apply(spec["grasp_point_local_m"])
    from pxr import Usd, UsdGeom
    from benchmark.catalog import object_asset

    stage = Usd.Stage.Open(str(object_asset(spec["type"])))
    root = stage.GetDefaultPrim()
    body = stage.GetPrimAtPath(f"{root.GetPath()}/{spec['contact_bodies'][0]}")
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"])
    grasp = body.GetChild(spec["grasp_prim"]) if "grasp_prim" in spec else body
    center = bounds.ComputeWorldBound(grasp).ComputeCentroid()
    local = UsdGeom.XformCache().GetLocalToWorldTransform(body).GetInverse().Transform(center)
    offset = np.array(local) * np.array(spec["scale"])
    if "grasp_point_local_m" in spec:
        offset = np.array(spec["grasp_point_local_m"])
    if spec["asset_kind"] == "articulation":
        i = asset.body_names.index(body.GetName())
        p, q = asset.data.body_pos_w[0, i], asset.data.body_quat_w[0, i]
    else:
        p, q = asset.data.root_pos_w[0], asset.data.root_quat_w[0]
    rotation = _rotation(q.cpu().tolist())
    return p.cpu().numpy() + rotation.apply(offset)


def interaction_path(env):
    """Nominal contact waypoints, without moving simulated task bodies."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    kind = env.task["episode_config"]["task_type"]
    if kind in {"wipe_bottle", "wipe_whiteboard"}:
        from benchmark.tasks.extra.wiping import interaction_path as wiping_path
        return wiping_path(env)
    if kind == "ram":
        from benchmark.tasks.extra.assembly import interaction_path as ram_path
        return ram_path(env)
    if kind == "surface_inspection":
        from benchmark.tasks.extra.inspection import interaction_path as inspection_path
        return inspection_path(env)
    if kind == "press_slide":
        from benchmark.tasks.extra.latch import interaction_path as latch_path
        return latch_path(env)
    if kind == "stand":
        from benchmark.tasks.extra.placement import interaction_path as standing_path
        return standing_path(env)
    if native_history_kind(kind):
        return contact_interaction_path(env)
    if kind == "slice":
        runtime = env.cutting
        aim, frame, _ = runtime.aim()
        normal = frame.apply([1,0,0])
        yaw = math.atan2(normal[1],normal[0])+math.pi/2
        tcp = env.scene["ee_frame"].data.target_pos_w[0,0].cpu().numpy()
        kp, kr = runtime.pose(runtime.knife)
        edge = kp+kr.apply(runtime.edge_local[abs(runtime.edge_local[:,0]-.1).argmin()])
        robot = env.scene["robot"]
        hand = _rotation(robot.data.body_quat_w[0,robot.body_names.index("panda_hand")].cpu().tolist())
        local = hand.inv().apply(edge-tcp)
        desired = Rotation.from_euler("z",yaw)*Rotation.from_euler("x",math.pi)
        offset = desired.apply(local)
        high = aim+np.array([0,0,.13])-offset
        low = aim.copy()
        low[2] = runtime.config["board_top_m"]+.001
        low -= offset
        milestones = [tcp, tcp+[0,0,.07], high, low, low+[0,0,.15]]
        return [(a+(b-a)*t,yaw) for a,b in zip(milestones,milestones[1:]) for t in np.linspace(0,1,8)]
    point = grasp_point(env)
    layout_q = env.reset_poses["mechanism"]["quaternion_wxyz"]
    layout = _rotation(layout_q)
    yaw = layout.as_euler("xyz")[2]
    if kind in {"drawer", "door"}:
        # Parallel jaws are pi-symmetric. Keep the approach on the nearby wrist
        # branch instead of commanding an unnecessary half-turn at a joint limit.
        yaw = (yaw+math.pi/2) % math.pi-math.pi/2
    points = [(point + [0, 0, 0.10], yaw), (point, yaw)]
    if kind == "drawer":
        for distance in np.linspace(0.02, 0.14, 8):
            points.append((point + layout.apply([distance, 0, 0]), yaw))
    elif kind in {"door", "valve", "switch"}:
        asset = env.scene["mechanism"]
        if kind == "door":
            index = asset.body_names.index("door_left_link")
            pivot = asset.data.body_pos_w[0, index].cpu().numpy()
            axis, angle = [0, 0, 1], -math.radians(66)
        elif kind == "valve":
            index = asset.body_names.index("Handle")
            pivot = asset.data.body_pos_w[0, index].cpu().numpy()
            axis, angle = [0, 0, 1], math.pi/2
        else:
            index = asset.body_names.index("Lever")
            pivot = asset.data.body_pos_w[0, index].cpu().numpy()
            axis, angle = layout.apply([0, 1, 0]), math.radians(40)
        for theta in np.linspace(0, angle, 25)[1:]:
            rot = Rotation.from_rotvec(np.asarray(axis)*theta)
            points.append((pivot + rot.apply(point-pivot), yaw+theta if kind in {"door", "valve"} else yaw))
    elif kind == "button":
        points.append((point + [0, 0, -0.010], yaw))
    elif kind == "cap":
        points.append((point + [0, 0, 0.06], yaw))
    elif kind == "contour":
        root = env.scene["mechanism"].data.root_pos_w[0].cpu().numpy()
        points = [(root+layout.apply([x, 0.004*math.sin(2*math.pi*x/0.16), 0.04]),
                   yaw+math.atan(0.004*2*math.pi/0.16*math.cos(2*math.pi*x/0.16)))
                  for x in np.linspace(-0.08, 0.08, 41)]
        points.insert(0, (root+layout.apply([-0.089, 0.004*2*math.pi/0.16*0.009, 0.04]), points[0][1]))
    else:
        base = env.scene["mechanism"].data.root_pos_w[0].cpu().numpy()
        anchor = base + layout.apply([-0.05, 0, 0])
        # Continue the hand arc beyond vertical if the object slips in the
        # pinch. Execution stops on the measured object angle, never its pose.
        for theta in np.linspace(0, math.radians(110), 38)[1:]:
            lever = 0.05+env.task["objects"]["mechanism"]["grasp_point_local_m"][0]
            p = anchor + layout.apply([lever*math.cos(theta)-0.01*math.sin(theta), 0,
                                       lever*math.sin(theta)+0.01*math.cos(theta)])
            points.append((p, yaw))
    return points


def contact_interaction_path(env):
    """Geometry-based nominal paths; native robot probes establish feasibility."""
    import numpy as np

    kind, goal = env.task["episode_config"]["task_type"], env.task["goals"][0]
    data = env.scene["mechanism"].data
    origin = data.root_pos_w[0].cpu().numpy()
    layout = _rotation(env.reset_poses["mechanism"]["quaternion_wxyz"])
    yaw = layout.as_euler("xyz")[2]
    point = grasp_point(env)
    points = [(point+[0, 0, .10], yaw), (point, yaw)]
    if kind == "push":
        config = env.task["episode_config"]["contact_path"]
        bounds = np.asarray(env.task["objects"]["mechanism"]["view_bounds_local_m"])
        half = (bounds[1]-bounds[0])/2
        points, normals = [], []
        # The round face lets the pad push along the curve tangent without
        # relying on lateral gel friction to steer a flat face.
        for u in np.linspace(0, 1, config.get("points", 81)):
            normal = layout.apply([config["length_m"],
                                   2*math.pi*config["amplitude_m"]*math.cos(2*math.pi*u), 0.])
            normal /= np.linalg.norm(normal)
            push_yaw = math.atan2(normal[1], normal[0])+math.pi/2
            center = origin+layout.apply([config["length_m"]*u,
                                         config["amplitude_m"]*math.sin(2*math.pi*u), 0.])
            # The authored affordance is the 25 mm high pushing contact.
            center[2] = point[2]
            points.append((center-half[0]*normal, push_yaw))
            normals.append(normal)
        points = single_pad_tcp_path(env, points, normals)
        points.insert(0, (points[0][0]+[0, 0, .08], points[0][1]))
    elif kind == "roll":
        # Side contact keeps both open jaws above the tabletop. Top contact
        # with the upper inner gel puts the opposite jaw below the cylinder.
        normal = layout.apply([1., 0., 0.])
        yaw = (yaw+math.pi) % math.pi-math.pi/2
        point = origin-layout.apply([goal["radius_m"], 0, 0])
        points = [(point+layout.apply([goal["radius_m"]*angle, 0, 0]), yaw)
                  for angle in np.linspace(0, goal["target_rad"], 81)]
        points = single_pad_tcp_path(env, points, normal)
        points.insert(0, (points[0][0]+[0, 0, .08], yaw))
    elif kind == "gears":
        points.extend((point, yaw+angle) for angle in np.linspace(0, goal["input_target_rad"], 25)[1:])
    elif kind == "hang":
        from benchmark.tasks.extra.contact_assets import asset_specs
        specs = asset_specs()
        mug, hook_spec = specs["extra_hanging_mug"], specs["extra_mug_hook"]
        hook = env.scene["hook"].data
        hook_rotation = _rotation(hook.root_quat_w[0].cpu().tolist())
        hook_position = hook.root_pos_w[0].cpu().numpy()
        seat = hook_position+hook_rotation.apply(hook_spec["seat_local_m"])
        tip = hook_position+hook_rotation.apply(hook_spec["tip_local_m"])
        # The 4 mm rim and the 7.6 mm handle tube are both thinner than the
        # ~10 mm closed gel gap, so the molded 13.5 mm grip slab at the ring's
        # Use the real raised grip-to-handle offset; the open ring carries
        # the cup only after it has passed over the hook's finite upturn.
        handle_local = np.asarray(mug["handle_center_local_m"])
        grasp_local = np.asarray(mug["handle_grip_center_local_m"])
        offset = handle_local-grasp_local
        target_yaw = hook_rotation.as_euler("xyz")[2]
        direction = hook_rotation.apply([0, 1, 0])
        entry = origin+layout.apply(grasp_local)+np.array([0, 0, .10])
        grasp = origin+layout.apply(grasp_local)
        raised = grasp+np.array([0, 0, .11])
        half_height = mug["handle_inner_ellipse_radii_xz_m"][1]
        # Pass the entire upturn through the hole and move past the tip.
        # The upper inner arc then lowers onto the shaft to carry the mug.
        # Clear the finite steel/tube radii plus the 3 mm TCP arrival window
        # before lowering, so the upturn cannot catch the upper handle arc.
        clearance = hook_spec['shaft_radius_m']+mug['handle_tube_radius_m']+.004
        cross_center = tip+direction*clearance-np.array([0., 0., .017])
        prethread_center = cross_center-direction*.10
        lower_center = np.array([cross_center[0], cross_center[1], seat[2]-half_height+.0032])
        points = [(p, yaw) for p in (entry, grasp, raised)]
        # Align the carried handle plane with the shaft before insertion;
        # retaining the pickup yaw can sweep the bowl into the hook.
        points.extend((p, target_yaw) for p in (
            prethread_center-hook_rotation.apply(offset),
            cross_center-hook_rotation.apply(offset), lower_center-hook_rotation.apply(offset),
            lower_center-hook_rotation.apply(offset)-direction*.07+[0, 0, .08]))
    elif kind == "slip":
        points.extend([(point+[0, 0, .10], yaw)]*2)
    return points


def single_pad_tcp_path(env, path, inward_normal):
    """Convert a physical surface path using measured open-jaw gel centers."""
    import numpy as np
    from benchmark.backends.gsmini.robot import _pad_centers_world

    robot = env.scene["robot"]
    hand = _rotation(robot.data.body_quat_w[0, robot.body_names.index("panda_hand")].cpu().tolist())
    tcp = env.scene["ee_frame"].data.target_pos_w[0, 0].cpu().numpy()
    pads = _pad_centers_world(env, ("gelpad_left", "gelpad_right"), env.cfg.gsmini_pad_centers_local)
    local = [hand.inv().apply(pad[0].cpu().numpy()-tcp) for pad in pads]
    result = []
    normals = np.asarray(inward_normal)
    if normals.shape == (3,):
        normals = np.broadcast_to(normals, (len(path), 3))
    if normals.shape != (len(path), 3) or not np.isfinite(normals).all():
        raise ValueError("One finite inward normal is required per surface waypoint")
    for (point, yaw), quaternion, normal in zip(path, path_quaternions(env, path), normals):
        rotation = _rotation(quaternion)
        offsets = [rotation.apply(offset) for offset in local]
        # Select the jaw whose inner gel surface faces the requested object.
        offset = min(offsets, key=lambda value: float(np.dot(value, normal)))
        result.append((point-normal*.0015-offset, yaw))
    return result


def contact_obstacle_paths(env, *, free_motion=False, payload=False):
    """Keep real geometry; only the current intended contact body is omitted.

    Free motion includes the target until a measured grasp attaches it as a
    payload. Gear outputs are never robot contact surfaces. Existing 0--9
    contact-body exclusions retain their previous meaning.
    """
    obstacles = ["/World/envs/env_0/Table"]
    kind = env.task["episode_config"]["task_type"]
    payload_name = "knife" if getattr(env, "cutting", None) is not None else "mechanism"
    for name, spec in env.task["objects"].items():
        if spec.get("reset_frame") == "robot_hand" or (payload and name == payload_name):
            continue
        if spec["asset_kind"] == "articulation":
            allowed = () if free_motion else (("Plunger",) if kind == "press_slide" and name == "mechanism"
                                               else ("Medium",) if kind == "gears" and name == "mechanism"
                                               else spec["contact_bodies"])
            obstacles.extend(f"/World/envs/env_0/{name}/{body}" for body in env.scene[name].body_names
                             if body not in allowed)
        elif free_motion or (name != "mechanism" and kind != "surface_inspection"):
            obstacles.append(f"/World/envs/env_0/{name}")
    return obstacles


def check_reachability(env):
    """Check sampled IK; report separately what continuous collision work remains."""
    import numpy as np
    import torch
    from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
    from curobo.types import DeviceCfg, GoalToolPose
    from benchmark.backends.gsmini.workspace import grasp_poses, read_collision_world
    from benchmark.backends.gsmini.sensors import planner_robot

    path = interaction_path(env)
    if env.task["episode_config"]["task_type"] == "contour":
        path, normals = contour_tcp_path(env, path)
        first, yaw = path[0]
        path = [(first-normals[0]*0.011+[0,0,0.08], yaw), (first-normals[0]*0.001,yaw), *path]
    if env.task["episode_config"]["task_type"] == "door":
        last, yaw = path[-1]
        path.append((last+[0, 0, 0.08], yaw))  # Open fingers, then retreat after opening.
    # All contact phases permit only the intended target bodies. Include the
    # table and each other link as real obstacles, rather than disabling collisions.
    obstacles = contact_obstacle_paths(env)
    world = read_collision_world(env, obstacle_paths=tuple(obstacles))
    solver = InverseKinematics(InverseKinematicsCfg.create(
        robot=planner_robot(env), scene_model=world, device_cfg=DeviceCfg(device=env.device),
        random_seed=env.seed_value, max_batch_size=max(64, len(path))))
    rows = []
    try:
        positions = torch.as_tensor(np.stack([position for position, _ in path]), device=env.device, dtype=torch.float32)
        quaternions = path_quaternions(env, path)
        pose = grasp_poses(env, positions, env_ids=slice(0,1), quaternion=quaternions)
        result = solver.solve_pose(GoalToolPose.from_poses({"panda_hand": pose}))
        valid = result.success.reshape(len(path), -1).any(dim=-1).cpu().tolist()
        rows = [{"waypoint": i, "position_m": path[i][0].tolist(), "reachable": ok} for i, ok in enumerate(valid)]
    finally:
        solver.destroy()
    report = {"passed": all(row["reachable"] for row in rows), "waypoints": rows,
              "scope": "Collision-aware sampled hand IK against the listed static obstacles",
              "obstacles": obstacles, "continuous_trajectory": {"passed": False, "status": "not_run"},
              "payload_collision": {"passed": False, "status": "not_required"},
              "collision_limitations": "Obstacles are a native pose snapshot; moving gear sweeps and intentional contact require native execution"}
    if native_history_kind(env.task["episode_config"]["task_type"]):
        report.update(check_continuous_reachability(env, path, obstacles))
        report["passed"] &= report["continuous_trajectory"]["passed"]
        if env.task["episode_config"]["task_type"] in {"hang", "stand"}:
            report["passed"] &= report["payload_collision"]["passed"]
    return report


def check_continuous_reachability(env, path, obstacles):
    """Plan connected joint trajectories; attach the mug during free transport.

    Intended contact bodies are omitted from the robot obstacle world. The mug
    is sphere-fit during transport, then native probes must validate hook contact.
    This diagnostic does not certify exact geometry or arbitrary executed paths.
    """
    import numpy as np
    import torch
    from curobo.types import GoalToolPose, JointState, Pose
    from curobo.sphere_fit import SphereFitType
    from benchmark.backends.gsmini.workspace import grasp_poses, read_collision_world
    from benchmark.backends.gsmini.motion import _attach_payload_spheres, _create_motion_planner, _planner_joint_state, _plan_nearest_branch

    kind = env.task["episode_config"]["task_type"]
    world = read_collision_world(env, obstacle_paths=tuple(obstacles))
    from benchmark.backends.gsmini.sensors import planner_robot
    planner = _create_motion_planner(env, env.seed_value, world,
        robot_config=lambda value: planner_robot(value, attached_sphere_count=16))
    segments, payload = [], {"passed": kind not in {"hang", "stand"}, "status": "not_required"}
    try:
        current = _planner_joint_state(env.scene["robot"], planner)
        # Keep every contact waypoint: shortcuts across a curved path do not
        # establish continuous access to its intermediate contact configurations.
        quaternions = path_quaternions(env, path)
        for index, (position, _) in enumerate(path):
            target = torch.as_tensor(np.asarray(position)[None], device=env.device, dtype=torch.float32)
            pose = grasp_poses(env, target, env_ids=slice(0, 1), quaternion=[quaternions[index]])
            plan = _plan_nearest_branch(planner, GoalToolPose.from_poses({"panda_hand": pose}), current)
            ok = plan is not None and bool(plan.success.all())
            segments.append({"waypoint": index, "passed": ok})
            if not ok:
                break
            trajectory = plan.get_interpolated_plan().reorder(planner.joint_names)
            positions = trajectory.position.reshape(-1, len(planner.joint_names))
            if not torch.isfinite(positions).all() or not len(positions):
                raise RuntimeError("Invalid continuous collision trajectory")
            current = JointState.from_position(positions[-1:].clone(), joint_names=planner.joint_names)
            segments[-1]["interpolated_steps"] = len(positions)
            if kind in {"hang", "stand"} and index == 1:
                geometry = read_collision_world(env, obstacle_paths=("/World/envs/env_0/mechanism",))
                if not geometry.objects:
                    raise RuntimeError("Missing attached task-object collision geometry")
                manager = planner.attachment_manager
                _attach_payload_spheres(manager, manager.attach, current, list(geometry.objects),
                    num_spheres=16, sphere_fit_type=SphereFitType.VOXEL,
                    world_objects_pose_offset=Pose.from_list([0, 0, 0, 1, 0, 0, 0], planner.device_cfg))
                payload = {"passed": False, "status": "transport_in_progress", "sphere_count": 16,
                           "geometry": [item.name for item in geometry.objects]}
            if (kind == "hang" and index == 3) or (kind == "stand" and index == 4):
                planner.attachment_manager.detach()
                payload.update(passed=True, status="free_transport_checked",
                               contact_phase="Hook engagement and lowering require native robot probes",
                               limitation="Sphere fit approximates mug geometry; exact handle-hole clearance is not certified")
                if kind == "stand":
                    payload.update(contact_phase="Native probes must establish controlled end-cap seating and release",
                                   limitation="Sphere fit bounds nominal rod transport; native contact and load history remain required")
    except Exception:
        return {"continuous_trajectory": {"passed": False, "segments": segments, "error": traceback.format_exc()},
                "payload_collision": payload}
    finally:
        planner.destroy()
    return {"continuous_trajectory": {"passed": len(segments) == len(path) and all(s["passed"] for s in segments),
            "segments": segments, "scope": "Connected collision-checked trajectories through every nominal contact waypoint"},
            "payload_collision": payload}


def contour_tcp_path(env, path):
    import numpy as np
    from benchmark.backends.gsmini.robot import _pad_centers_world
    pads = _pad_centers_world(env, ("gelpad_left", "gelpad_right"), env.cfg.gsmini_pad_centers_local)
    tcp = env.scene["ee_frame"].data.target_pos_w[0,0].cpu().numpy()
    hand = env.scene["robot"]
    hi = hand.body_names.index("panda_hand")
    local = _rotation(hand.data.body_quat_w[0,hi].cpu().tolist()).inv().apply(pads[0][0].cpu().numpy()-tcp)
    targets, normals = [], []
    for point, yaw in path:
        desired = _rotation([0, math.cos(yaw/2), math.sin(yaw/2), 0])
        offset = desired.apply(local)
        normal = offset/np.linalg.norm(offset)
        # Pad centroids are inside a 4 mm gel. Place its front on the near
        # side of the 6 mm rail, rather than driving the centroid into the rail.
        targets.append((point-offset+normal*0.0055, yaw))
        # The inner gel face points back toward the TCP. Advancing along the
        # TCP-to-pad offset instead unloads the contact at every force update.
        normals.append(-normal)
    return targets, normals


def path_quaternions(env, path):
    from scipy.spatial.transform import Rotation
    quaternions = []
    task_type = env.task["episode_config"]["task_type"]
    switch = task_type == "switch"
    initial = env.initial_state.get("joint_position", 0.0)
    for i, (_, yaw) in enumerate(path):
        if task_type == "hang":
            # Top-down handle-tube pinch: close across the handle loop (the
            # mug's Y axis) so both sensor cases stay clear of the body.
            rotation = Rotation.from_euler("z", yaw)*Rotation.from_euler("x", math.pi)
        else:
            tilt = initial + math.radians(40)*max(0,i-1)/max(1,len(path)-2) if switch else 0.0
            if task_type == "stand" and i >= 3:
                tilt = -math.pi/2
            rotation = Rotation.from_euler("z", yaw)*Rotation.from_euler("y", tilt)
            if switch:
                rotation *= Rotation.from_euler("z", -math.pi/2)
            rotation *= Rotation.from_euler("x", math.pi)
        x,y,z,w = rotation.as_quat()
        quaternions.append([w,x,y,z])
    return quaternions


def tcp_pose_error(env, position, quaternion):
    """Measured world TCP error, including antipodal quaternion equivalence."""
    import torch

    data = env.scene["ee_frame"].data
    target = torch.as_tensor(position, device=data.target_pos_w.device, dtype=data.target_pos_w.dtype)
    distance = float(torch.linalg.vector_norm(data.target_pos_w[:, 0]-target, dim=-1).max())
    current = data.target_quat_w[:, 0]
    desired = current.new_tensor(quaternion).reshape(-1, 4)
    if (not torch.isfinite(data.target_pos_w[:, 0]).all() or not torch.isfinite(target).all()
            or not torch.isfinite(current).all() or not torch.isfinite(desired).all()
            or (current.norm(dim=-1) <= 0).any() or (desired.norm(dim=-1) <= 0).any()):
        raise RuntimeError("Invalid measured or desired TCP pose")
    cosine = (current*desired).sum(-1).abs()/(current.norm(dim=-1)*desired.norm(dim=-1))
    angle = float((2*torch.acos(cosine.clamp(0., 1.))).max())
    return distance, angle


class _MotionTrajectoryEnvironment:
    """Let the shared trajectory executor retain this motion's checked steps."""

    def __init__(self, motion):
        self.motion = motion

    def __getattr__(self, name):
        return getattr(self.motion.env, name)

    def step(self, action):
        import torch

        if action is not self.motion.env.action:
            raise RuntimeError("Contact trajectory must use the recorded robot action")
        self.motion.step()
        # ExtraEnvironment returns task success separately; motion.step handles
        # its timeout/invalid-state guards before returning to this adapter.
        flags = torch.zeros(action.shape[0], dtype=torch.bool, device=action.device)
        return None, None, flags, flags, {}


class ContactMotion:
    """Bounded live-Jacobian motion using the collector's step implementation.

    Never writes any body's pose or joint state.
    """

    def __init__(self, env):
        from benchmark.backends.gsmini.motion import _make_pose_controller
        self.env = env
        self.controller = _make_pose_controller(env)
        self.pivoting = False
        self.support_correction_m = 0.0
        self.surface_normal = None
        self.normal_correction_m = 0.0
        self.normal_correction_limit_m = 0.001
        self.contact_force_target_n = 0.1
        self.contact_force_limit_n = None
        self.previous_grip_gap_z = None

    def _prepare_step(self):
        """Apply shared contact feedback before the next robot step."""
        if getattr(self, "cup_carry_servo", False) and self.env.expert_phase in (1, 2):
            from benchmark.tasks.extra import gsmini_contact as extra_contact
            state = extra_contact._state(self.env)
            extra_contact._housing_load(state)
            if min(state["left_gel_force_n"], state["right_gel_force_n"]) < extra_contact._carrying_preload(self.env):
                self.env.action[:, -1] = max(self.cup_closure_floor, float(self.env.action[0, -1])-.001)

    def hold(self, duration):
        for _ in range(math.ceil(duration/self.env.step_dt)):
            self.step()

    def move_planned(self, position, yaw, *, duration=1., quaternion=None,
                     converge=True, rotation_tolerance=.05, payload=False, repeat_factor=1):
        """Execute cuRobo's joint samples; hold that endpoint for pose arrival.

        ``duration`` is accepted for compatibility with ``move``; trajectory
        timing follows the planned joint samples and ``repeat_factor``.
        This covers free approach/transport against measured obstacle geometry.
        An attached object uses 16 fitted spheres. Local contact servo and
        moving obstacle sweeps remain outside this collision-planning scope.
        """
        import torch
        from benchmark.backends.gsmini.workspace import grasp_poses, read_collision_world
        from benchmark.backends.gsmini.sensors import planner_robot
        from benchmark.backends.gsmini.motion import _attach_payload_spheres, _create_motion_planner, _planner_joint_state, _plan_pose

        env = self.env
        cutting_payload = payload and getattr(env, "cutting", None) is not None
        payload_name = "knife" if cutting_payload else "mechanism"
        q = quaternion if quaternion is not None else [0, math.cos(yaw/2), math.sin(yaw/2), 0]
        if not converge or rotation_tolerance is None or not 0 < rotation_tolerance < math.pi:
            raise ValueError("Planned contact motion requires position and rotation arrival checks")
        obstacles = contact_obstacle_paths(env, free_motion=True, payload=payload)
        row = {"status": "planning", "payload": payload, "obstacles": obstacles,
                "scope": "Executed free-motion joint plan against measured static obstacles; contact servo and moving sweeps are not certified"}
        if not hasattr(self, "planned_segments"):
            self.planned_segments = []
        self.planned_segments.append(row)
        planner, attached = None, False
        try:
            world = read_collision_world(env, obstacle_paths=tuple(obstacles))
            def robot_config(value):
                config = planner_robot(value, attached_sphere_count=16 if payload else 0)
                if cutting_payload:
                    robot = value.scene["robot"]
                    locks = config["robot_cfg"]["kinematics"]["lock_joints"]
                    for name in locks:
                        locks[name] = float(robot.data.joint_pos[0, robot.joint_names.index(name)])
                return config
            planner = _create_motion_planner(env, env.seed_value, world,
                robot_config=robot_config, **({"trajectory_knots": 6} if cutting_payload or env.task["episode_config"]["task_type"] in {"wipe_whiteboard","wipe_bottle"} else {}))
            if payload:
                from curobo.types import Pose
                from curobo.sphere_fit import SphereFitType

                geometry = read_collision_world(env, obstacle_paths=(f"/World/envs/env_0/{payload_name}",))
                if not geometry.objects:
                    raise RuntimeError("Missing measured attached-object collision geometry")
                manager = planner.attachment_manager
                attached = True  # Detach even if fitting fails after partial attachment.
                _attach_payload_spheres(manager, manager.attach,
                    _planner_joint_state(env.scene["robot"], planner), list(geometry.objects),
                    num_spheres=16, sphere_fit_type=SphereFitType.VOXEL,
                    world_objects_pose_offset=Pose.from_list([0, 0, 0, 1, 0, 0, 0], planner.device_cfg))
                row["payload_geometry"] = [item.name for item in geometry.objects]
                row["payload_sphere_count"] = 16
                row["payload_limitation"] = "VOXEL sphere fit approximates geometry; exact handle-hole clearance is not certified"
            target = torch.as_tensor(position, device=env.device, dtype=torch.float32).reshape(1, 3)
            goal = grasp_poses(env, target, quaternion=q)
            if payload and env.task["episode_config"]["task_type"] == "slip":
                from curobo.types import ToolPoseCriteria
                planner.update_tool_pose_criteria({"panda_hand": ToolPoseCriteria.linear_motion(
                    axis="z", non_terminal_scale=10., project_distance_to_goal=False)})
            app = getattr(self, "app", None)
            if app is None:
                import omni.kit.app
                app = omni.kit.app.get_app()
            reached, distance, angle = 0, math.inf, math.inf
            arrival_during_plan=cutting_payload and env.task["episode_config"].get("food") in {"banana","baguette_1","parsnip","potato","zucchini_1"}
            original_step=self.step
            def checked_step():
                nonlocal reached,distance,angle
                result=original_step()
                distance,angle=tcp_pose_error(env,target,q)
                reached=reached+1 if distance<.003 and angle<rotation_tolerance else 0
                return result
            if arrival_during_plan:
                self.step=checked_step
            try:
                _plan_pose(_MotionTrajectoryEnvironment(self),app,planner,env.action,goal,
                           "contact_transport" if payload else "contact_approach",
                           settle_seconds=0. if cutting_payload and env.task["episode_config"].get("food") in {"carrot","cucumber_7","banana","baguette_1","parsnip","potato","zucchini_1"} else .1,
                           repeat_factor=repeat_factor,interpolate_repeats=payload)
            finally:
                self.step=original_step
            # Keep the collision-checked endpoint. The complete plan can earn
            # the same two fresh arrival samples during its final steps.
            for _ in range(max(2,math.ceil(2./env.step_dt))):
                if reached>=2:
                    break
                checked_step()
            if reached < 2:
                raise RuntimeError(f"Planned robot motion did not reach its physical target: error={distance:.4f} m, angle={angle:.4f} rad")
            row.update(status="executed", position_error_m=distance, rotation_error_rad=angle)
        except Exception as exc:
            row.update(status="failed", error=str(exc))
            raise
        finally:
            if planner is not None:
                try:
                    if attached:
                        planner.attachment_manager.detach()
                finally:
                    planner.destroy()

    def move(self, position, yaw, *, duration=1.0, quaternion=None, converge=False, rotation_tolerance=None,
             until=None, interpolate=False, center_joints=True):
        """Servo for the budget, or until two consecutive contact feedback samples agree."""
        import torch
        from benchmark.backends.gsmini.workspace import grasp_poses
        from benchmark.backends.gsmini.motion import _hand_pose_joint_target

        env = self.env
        q = quaternion if quaternion is not None else [0, math.cos(yaw/2), math.sin(yaw/2), 0]
        positions = torch.tensor([list(position)], device=env.device, dtype=torch.float32)
        minimum = math.ceil(duration/env.step_dt)
        budget = max(minimum, math.ceil(6.0/env.step_dt)) if converge else minimum
        if interpolate:
            from scipy.spatial.transform import Rotation, Slerp

            start_position = env.scene["ee_frame"].data.target_pos_w[:,0].clone()
            start_quaternion = env.scene["ee_frame"].data.target_quat_w[0,0].cpu().numpy()
            orientations = Rotation.from_quat([start_quaternion[[1,2,3,0]],
                                               [q[1],q[2],q[3],q[0]]])
            rotation_path = Slerp([0.,1.],orientations)
        error, angle, reached, confirmed = math.inf, math.inf, 0, 0
        for index in range(budget):
            target_position = positions.clone()
            target_quaternion = q
            if interpolate:
                fraction = min(1.,(index+1)/minimum)
                fraction = fraction**3*(10.-15.*fraction+6.*fraction**2)
                target_position = start_position+(positions-start_position)*fraction
                target_quaternion = rotation_path(fraction).as_quat()[[3,0,1,2]].tolist()
            if self.pivoting:
                target_position[:,2] += self.support_correction_m
            if self.surface_normal is not None:
                target_position += target_position.new_tensor(self.surface_normal)*self.normal_correction_m
            if (getattr(env, "expert_phase", 0) >= 1
                    and env.task["episode_config"]["task_type"] == "press_slide"):
                # Break static shear indeterminacy through recorded robot motion.
                layout = _rotation(env.reset_poses["mechanism"]["quaternion_wxyz"])
                transverse = .0002*math.sin(2*math.pi*4*env.extra_step_counter*env.step_dt)
                target_position += target_position.new_tensor(layout.apply([0., transverse, 0.]))
            goal = grasp_poses(env, target_position, quaternion=target_quaternion)
            arm = env.action_manager.get_term("arm_action")
            # Curved sliding contact needs enough wrist response to keep its
            # normal aligned. Free approach retains the smaller IK increment.
            max_step = .04
            if (getattr(env, "expert_phase", 0) >= 2
                    and env.task["episode_config"]["task_type"] == "push"):
                max_step = .12 if env.progress.furthest_u < .5 else .16
            target = _hand_pose_joint_target(env, self.controller, goal,
                                            max_joint_step=max_step, center_joints=center_joints)
            env.action[:, :arm.action_dim] = arm.position_actions(target)
            result = self.step()
            error = float(torch.linalg.vector_norm(env.scene["ee_frame"].data.target_pos_w[:,0]-target_position))
            if rotation_tolerance is not None:
                error, angle = tcp_pose_error(env, target_position, target_quaternion)
            if converge:
                arrived = error < 0.003 and (rotation_tolerance is None or angle < rotation_tolerance)
                reached = reached+1 if arrived else 0
                if index+1 >= minimum and reached >= (2 if rotation_tolerance is not None else 1):
                    break
            if self.surface_normal is not None:
                force = result["metrics"]["gel_force_n"]
                # Positive correction advances into the surface.
                target = self.contact_force_target_n
                upper = self.contact_force_limit_n if self.contact_force_limit_n is not None else 2*target
                delta = .0001 if force < target else -.0001 if force > upper else 0.
                self.normal_correction_m = max(-0.003, min(self.normal_correction_limit_m,
                                                           self.normal_correction_m+delta))
            if self.pivoting:
                from benchmark.expert.tabletop import _peg_gripper_command
                state = result["metrics"]
                gap = grasp_point(env)-env.scene["ee_frame"].data.target_pos_w[0,0].cpu().numpy()
                speed = 0.0 if self.previous_grip_gap_z is None else (gap[2]-self.previous_grip_gap_z)/env.step_dt
                self.previous_grip_gap_z = gap[2]
                if sum(g*g for g in gap) > 0.025**2:
                    raise RuntimeError("Pivot escaped the controlled pinch")
                command = _peg_gripper_command(float(env.action[0,-1]), state["robot_force_n"]/2, 0.45, speed)
                env.action[:, -1] = command
                if state["table_force_n"] < 0.05:
                    self.support_correction_m = max(-0.012, self.support_correction_m-0.0005)
                elif state["table_force_n"] > 1.0:
                    self.support_correction_m = min(0.0, self.support_correction_m+0.0002)
                if state["pivot_rad"] >= math.radians(89.8):
                    from benchmark.backends.gsmini.environment import _hold_current_arm_action
                    _hold_current_arm_action(env, env.action)
                    break
            if until is not None:
                confirmed = confirmed+1 if until(result) else 0
                if confirmed >= 2:
                    break
        if (converge or rotation_tolerance is not None) and (
                not math.isfinite(error) or error >= 0.003
                or (rotation_tolerance is not None and (not math.isfinite(angle) or angle >= rotation_tolerance))):
            raise RuntimeError(f"Robot approach did not reach its physical target: error={error:.4f} m, angle={angle:.4f} rad; target={position}")


def inspect_reset(env):
    """Check native PhysX mechanics and the actual table collision surface."""
    import numpy as np
    import torch
    import omni.physx
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade
    from isaaclab.sim import get_current_stage

    errors, assets = [], {}
    for name, spec in env.task["objects"].items():
        asset = env.scene[name]
        view = asset.root_physx_view
        mass = view.get_masses().cpu().numpy()
        inertia = view.get_inertias().cpu().numpy().reshape(-1,3,3)
        if not np.isfinite(mass).all() or (mass <= 0).any():
            errors.append(f"{name}: invalid native mass")
        if not np.isfinite(inertia).all() or (np.linalg.eigvalsh(inertia) <= 0).any():
            errors.append(f"{name}: invalid native inertia")
        root = asset.data.root_pos_w[0].cpu().numpy()
        if spec["asset_kind"] == "articulation" or spec.get("kinematic") or spec.get("fixed_mount"):
            requested = np.array(env.reset_poses[name]["position"])
            if np.linalg.norm(root-requested) > 0.001:
                errors.append(f"{name}: fixed base did not retain its seeded transform")
        row = {"mass_kg": mass.tolist(), "inertia_kg_m2": inertia.tolist(), "root_position_m": root.tolist()}
        row.update(native_contact_offsets_m=view.get_contact_offsets().cpu().tolist(),
                   native_rest_offsets_m=view.get_rest_offsets().cpu().tolist(),
                   native_shape_materials=view.get_material_properties().cpu().tolist(),
                   gravity_disabled=view.get_disable_gravities().cpu().tolist())
        if np.any(row["gravity_disabled"]):
            errors.append(f"{name}: task-body gravity is disabled")
        if spec["asset_kind"] == "rigid" and not spec.get("kinematic"):
            linear = float(torch.linalg.vector_norm(asset.data.root_lin_vel_w[0]))
            angular = float(torch.linalg.vector_norm(asset.data.root_ang_vel_w[0]))
            row["settling"] = {"linear_speed_m_s": linear, "angular_speed_rad_s": angular,
                               "max_linear_speed_m_s": .01, "max_angular_speed_rad_s": .1,
                               "passed": math.isfinite(linear) and math.isfinite(angular) and linear <= .01 and angular <= .1}
            if not row["settling"]["passed"]:
                errors.append(f"{name}: free rigid body has not settled")
        if spec["asset_kind"] == "articulation":
            q = asset.data.joint_pos
            limits = view.get_dof_limits().to(q.device)
            if ((q < limits[...,0]-0.001) | (q > limits[...,1]+0.001)).any():
                errors.append(f"{name}: joint limit violation")
            if asset.data.joint_vel.abs().max() > 0.02:
                errors.append(f"{name}: mechanism is not at rest")
            row.update(joints=asset.joint_names, joint_positions=q.cpu().tolist(),
                       joint_velocities=asset.data.joint_vel.cpu().tolist(), limits=limits.cpu().tolist(),
                       joint_stiffness_si=view.get_dof_stiffnesses().cpu().tolist(),
                       joint_damping_si=view.get_dof_dampings().cpu().tolist())
            for joint, passive in spec.get("passive_actuators", {}).items():
                j = asset.joint_names.index(joint)
                for label in ("stiffness", "damping"):
                    if not math.isclose(row[f"joint_{label}_si"][0][j], passive[label], rel_tol=1e-5, abs_tol=1e-7):
                        errors.append(f"{name}/{joint}: native {label} differs from SI task parameters")
        assets[name] = row
    state = env.read_task_state()
    if env.task["episode_config"]["task_type"] in {"drawer", "door"} and not state["cabinet_intact"]:
        errors.append("cabinet geometry, hinge/handle attachment or world mount is invalid")
    if state["robot_force_n"] >= env.task["success_config"]["contact_threshold_n"]:
        errors.append("robot is already in contact at reset")
    if env.progress.result["success"]:
        errors.append("task is already successful at reset")
    # Raycast only the table; visual bounds include its raised edge.
    table_hits = []
    def hit(item):
        if "/Table/" in item.collision:
            table_hits.append(float(item.position[2]))
        return True
    position = env.reset_poses["mechanism"]["position"]
    omni.physx.get_physx_scene_query_interface().raycast_all((position[0],position[1],1.0), (0,0,-1), 1.0, hit)
    if not table_hits:
        errors.append("missing physical tabletop at reset position")
    stage = get_current_stage()
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"])
    transforms = UsdGeom.XformCache()
    for name, spec in env.task["objects"].items():
        root = stage.GetPrimAtPath(f"/World/envs/env_0/{name}")
        asset = env.scene[name]
        colliders = []
        for prim in Usd.PrimRange(root):
            if not prim.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                continue
            body = prim
            while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
                body = body.GetParent()
            if not body:
                errors.append(f"{prim.GetPath()}: collision has no rigid owner")
                continue
            index = asset.body_names.index(body.GetName())
            rotation = _rotation(asset.data.body_quat_w[0,index].cpu().tolist())
            position = asset.data.body_pos_w[0,index].cpu().numpy()
            local_bounds = bounds.ComputeUntransformedBound(prim).ComputeAlignedRange()
            if local_bounds.IsEmpty():
                errors.append(f"{prim.GetPath()}: empty collision bounds")
                continue
            lower, upper = local_bounds.GetMin(), local_bounds.GetMax()
            matrix = transforms.GetLocalToWorldTransform(prim)*transforms.GetLocalToWorldTransform(body).GetInverse()
            corners = [matrix.Transform(Gf.Vec3d(x,y,z)) for x in (lower[0],upper[0])
                       for y in (lower[1],upper[1]) for z in (lower[2],upper[2])]
            minimum_z = float((rotation.apply(np.array(corners))+position)[:,2].min())
            support_geometry = {}
            if spec["type"] == "extra_roll_can" and prim.GetName() == "Collision":
                from benchmark.tasks.extra.contact_assets import roll_collision_support

                minimum_z, support_geometry = roll_collision_support(prim, matrix, rotation, position, spec["scale"])
                if not support_geometry["matches_declared_geometry"]:
                    errors.append(f"{prim.GetPath()}: cylinder support differs from collision geometry")
            if table_hits and minimum_z < max(table_hits)-0.001:
                errors.append(f"{prim.GetPath()}: penetrates tabletop by {max(table_hits)-minimum_z:.6f} m")
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
            physics = UsdPhysics.MaterialAPI(material.GetPrim()) if material else None
            colliders.append({"path": str(prim.GetPath()), "body": body.GetName(), "minimum_z_m": minimum_z,
                **support_geometry,
                "contact_offset_m": prim.GetAttribute("physxCollision:contactOffset").Get(),
                "rest_offset_m": prim.GetAttribute("physxCollision:restOffset").Get(),
                "static_friction": physics.GetStaticFrictionAttr().Get() if physics else None,
                "dynamic_friction": physics.GetDynamicFrictionAttr().Get() if physics else None,
                "restitution": physics.GetRestitutionAttr().Get() if physics else None,
                "friction_combine": material.GetPrim().GetAttribute("physxMaterial:frictionCombineMode").Get() if material else None})
        if not colliders:
            errors.append(f"{name}: no enabled native colliders")
        assets[name]["colliders"] = colliders
    robot = env.scene["robot"].root_physx_view
    robot_contact = {"material_columns": ["static_friction","dynamic_friction","restitution_or_negative_compliant_stiffness_n_m"],
        "compliant_contact_note": "PhysX overloads negative restitution as spring stiffness; the delivered GSmini gel uses 10 N/m and 1 N s/m damping.",
        "native_shape_materials": robot.get_material_properties().cpu().tolist(),
        "native_contact_offsets_m": robot.get_contact_offsets().cpu().tolist(),
        "native_rest_offsets_m": robot.get_rest_offsets().cpu().tolist()}
    penetration = {"passed": False, "status": "not_measured",
                   "scope": "Only collider support bounds versus the physical tabletop were checked"}
    runtime = getattr(env, "contact_runtime", None)
    if runtime is not None:
        penetration = runtime.penetration_report()
        if not penetration["passed"]:
            errors.append("native initial contact penetration exceeds the reset tolerance")
    return {"passed": not errors, "errors": errors, "assets": assets, "robot_contacts": robot_contact,
            "native_initial_penetration": penetration,
            "table_surface_m": max(table_hits) if table_hits else None, "state": state}
