"""Shared robot motion planning and grasp verification helpers."""

from __future__ import annotations

from benchmark.backends.gsmini.environment import PlanningFailed


def _collision_cache_sizes(*worlds):
    """Fit every planned world without launching collision lanes for spare slots."""
    return {
        kind: max((len(getattr(world, kind) or ()) for world in worlds), default=0)
        for kind in ("mesh", "cuboid")
    }


def _create_motion_planner(env, seed: int, world, *cache_worlds,
                           robot_config=None, velocity_scale: float = 1.0,
                           num_trajopt_seeds: int | None = None,
                           trajectory_knots: int | None = None):
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import DeviceCfg
    if robot_config is None:
        from benchmark.backends.gsmini.sensors import planner_robot
        robot_config = planner_robot
    if not 0.0 < velocity_scale <= 1.0:
        raise ValueError("velocity_scale must be in (0, 1]")
    robot = robot_config(env)
    if velocity_scale != 1.0:
        robot_cfg = robot.get("robot_cfg", robot)
        robot_cfg["kinematics"]["cspace"]["velocity_scale"] = velocity_scale

    options = {}
    if trajectory_knots is not None:
        from curobo.content import get_task_configs_path
        from curobo.config_io import load_yaml

        if type(trajectory_knots) is not int or trajectory_knots < 6:
            raise ValueError("trajectory_knots must be an integer >= 6")
        # cuRobo rounds each spline span to a control tick. Fewer knots let
        # short moves finish sooner without resampling or skipping validation.
        transition = load_yaml(str(get_task_configs_path() / "trajopt/transition_bspline_trajopt.yml"))
        transition["transition_model_cfg"]["n_knots"] = trajectory_knots
        options["trajopt_transition_model"] = transition

    planner = MotionPlanner(MotionPlannerCfg.create(
        robot=robot, scene_model=world,
        collision_cache=_collision_cache_sizes(world, *cache_worlds),
        device_cfg=DeviceCfg(device=env.device),
        interpolation_dt=env.step_dt,
        random_seed=seed, max_batch_size=1,
        num_trajopt_seeds=num_trajopt_seeds,
        **options,
    ))
    planner._collision_geometry_version = getattr(world, "_geometry_version", None)
    return planner


def _planner_joint_state(robot, planner):
    """Snapshot measured joints in cuRobo's order without aliasing PhysX buffers."""
    from curobo.types import JointState

    return JointState.from_position(
        robot.data.joint_pos.clone(), joint_names=robot.joint_names,
    ).reorder(planner.joint_names)


def _attach_payload_spheres(manager, attach, *args, **kwargs):
    """Preserve ordinary fits; conservatively enclose a failed VOXEL payload fit."""
    from unittest.mock import patch

    original_fit = manager.fit_spheres

    def fit_spheres(*fit_args, **fit_kwargs):
        spheres = original_fit(*fit_args, **fit_kwargs)
        result = getattr(manager, "_last_fit_result", None)
        diagnostics = getattr(result, "debug_info", None) or {}
        if diagnostics.get("fit_type") == "voxel" and diagnostics.get("fallback_used"):
            import numpy as np
            import torch

            # Pinned cuRobo retries VOXEL unchanged, then samples fixed-radius
            # surface spheres. Enclose its complete mesh (or convex hull) in
            # that same frame; attachment.update applies the link transform.
            vertices = np.asarray(getattr(getattr(result, "used_mesh", None), "vertices", []), dtype=np.float64)
            if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
                raise RuntimeError("Cannot conservatively enclose an invalid payload mesh")
            center = spheres.new_tensor((vertices.min(axis=0) + vertices.max(axis=0)) / 2)
            # Account for center rounding before rounding the radius upward.
            radius = np.linalg.norm(vertices - center.detach().cpu().numpy(), axis=1).max()
            if not np.isfinite(radius) or radius <= 0:
                raise RuntimeError("Cannot conservatively enclose a degenerate payload mesh")
            radius = torch.nextafter(spheres.new_tensor(radius), spheres.new_tensor(float("inf")))
            spheres = torch.cat((center, radius[None]))[None]
            print(f"[CUROBO] VOXEL payload fallback: enclosing {len(vertices)} mesh vertices "
                  f"with radius {radius.item():.6f}m", flush=True)
        if spheres.shape[0] != 1:
            return spheres
        # Use cuRobo's disabled-slot radius; keep the active collision sphere unchanged.
        padded = spheres.new_zeros((2, 4))
        padded[0] = spheres[0]
        padded[1, 3] = -100.0
        return padded

    with patch.object(manager, "fit_spheres", fit_spheres):
        return attach(*args, **kwargs)


def _attach_payload_for_planning(
    env, planner, obstacle_names, read_collision_world,
    object_name: str, excluded_obstacle_names=(),
    include_table=True,
    object_path: str | None = None,
):
    """Attach measured scene geometry with cuRobo after physical grasp checks."""
    from curobo.sphere_fit import SphereFitType
    from curobo.types import DeviceCfg, Pose
    from benchmark.backends.gsmini.workspace import update_collision_world

    obstacles = tuple(name for name in obstacle_names
                      if name not in excluded_obstacle_names)
    world_objects = tuple(dict.fromkeys(
        (("Table",) if include_table else ()) + obstacles + (object_name,)))
    world = read_collision_world(env, world_objects)
    update_collision_world(planner, world)
    payload_path = object_path or f"/World/envs/env_0/{object_name}"
    names = [item.name for item in world.objects
             if item.name == payload_path or item.name.startswith(payload_path + "/")]
    if not names:
        raise RuntimeError(
            f"cuRobo collision world is missing {object_name}; "
            f"available objects: {[item.name for item in world.objects]}"
        )
    robot = env.scene["robot"]
    current = _planner_joint_state(robot, planner)
    pose_offset = Pose.from_list([0, 0, 0, 1, 0, 0, 0], DeviceCfg(device=env.device))
    collision_names = set(planner.scene_collision_checker.get_obstacle_names())
    missing = set(names) - collision_names
    manager = planner.attachment_manager
    attach, payload, options = manager.attach_from_scene, names, {}
    if missing:
        payload = [world.get_obstacle(name) for name in names]
        if any(obstacle is None for obstacle in payload):
            raise RuntimeError(f"cuRobo scene lost payload geometry: {sorted(missing)}")
        attach = manager.attach
        options["disable_obstacle_names"] = [name for name in names if name in collision_names]
    _attach_payload_spheres(
        manager, attach, current, payload, num_spheres=4, sphere_fit_type=SphereFitType.VOXEL,
        # cuRobo already transformed scene meshes into the robot frame.
        world_objects_pose_offset=pose_offset, **options,
    )
    if missing:
        print(f"[CUROBO] payload absent from world cache; attaching directly: "
              f"{sorted(missing)}", flush=True)
    print(f"[CUROBO] attached payload collision: {names}", flush=True)


def _nearest_joint_solution(candidates, success, current, *, limits=None, joint_margin=0.0,
                            max_joint_jump=0.1):
    """Choose the successful IK solution nearest the current joints, rejecting branch jumps."""
    import torch

    delta = candidates - current[:, None, :]
    continuous = success.clone()
    if max_joint_jump is not None:
        continuous &= delta.abs().amax(dim=-1) <= max_joint_jump
    if limits is not None:
        clearance = torch.minimum(candidates - limits[:, None, :, 0],
                                  limits[:, None, :, 1] - candidates).amin(dim=-1)
        continuous &= clearance >= joint_margin
        best = clearance.masked_fill(~continuous, -torch.inf).amax(dim=-1, keepdim=True)
        continuous &= clearance >= best - 1e-6
    if not continuous.any(dim=-1).all():
        raise RuntimeError("Rejected an IK joint branch jump")
    distance = torch.linalg.vector_norm(delta, dim=-1).masked_fill(
        ~continuous, float("inf"))
    return candidates[torch.arange(current.shape[0], device=current.device),
                      distance.argmin(dim=-1)]


def _plan_nearest_branch(planner, goal, current):
    """Plan to a hand pose from the IK solution nearest the measured joints.

    cuRobo seeds trajectory optimization with arbitrary IK solutions, so the same hand
    pose can land on distant arm branches across seeds. Seeding from the nearest solution
    keeps the arm on its current branch while preserving pose goals and tool-pose criteria.
    """
    plan = None
    for attempt in range(5):  # cuRobo's default planning attempts.
        state = current.clone()
        ik = planner.ik_solver.solve_pose(goal, state, return_seeds=planner.ik_solver.config.num_seeds)
        success = ik.success.reshape(current.position.shape[0], -1)
        if not success.any(dim=-1).all():
            continue
        candidates = ik.solution.reshape(*success.shape, -1)
        nearest = _nearest_joint_solution(candidates, success, current.position, max_joint_jump=None)
        seeds = nearest[:, None].repeat(1, planner.trajopt_solver.config.num_seeds, 1)
        options = {}
        if attempt >= 1 and getattr(planner, "graph_planner", None) is not None:
            # Retain cuRobo's obstacle-routing fallback on the selected IK branch.
            trajectory = planner._get_graph_seed_trajectories(state, seeds)
            if trajectory is None:
                continue
            options.update(seed_traj=trajectory, finetune_attempts=3, finetune_dt_scale=0.75)
        plan = planner.trajopt_solver.solve_pose(
            goal, state, seed_config=seeds, use_implicit_goal=True, **options)
        if plan.success.any():
            break
    return plan


def _plan_pose(env, app, planner, action, hand_pose, label, *,
               allow_gripper_contact=False, repeat_factor=1,
               goal_for_duration=None, settle_seconds=None, interpolate_repeats=False):
    """Plan and replay a collision-aware hand pose through Isaac Lab actions."""
    import torch
    from curobo.types import GoalToolPose, Pose, ToolPoseCriteria
    from benchmark.backends.gsmini.execution import execute_trajectory

    robot = env.scene["robot"]
    current = _planner_joint_state(robot, planner)
    linear_insertion = label in ("insertion", "angled_insertion", "dynamic_insert",
                                 "vertical_insertion")
    if settle_seconds is None:
        settle_seconds = 1.0 if linear_insertion else getattr(env, "motion_free_settle_s", 0.2)
    if linear_insertion:
        non_terminal_scale = 1.0 if label == "vertical_insertion" else 10.0
        if label == "angled_insertion":
            non_terminal_scale = getattr(env, "insertion_path_scale", 1.0)
        # Goal-frame projection makes local z follow Task 0's tilted bore.
        criterion = ToolPoseCriteria.linear_motion(
            axis="z", non_terminal_scale=non_terminal_scale,
            project_distance_to_goal=label == "angled_insertion")
        planner.update_tool_pose_criteria({"panda_hand": criterion})
    contact_links = ()
    if allow_gripper_contact:
        contact_links = planner.kinematics.config.kinematics_config.grasp_contact_link_names or ()
        planner.disable_link_collision(contact_links)
    try:
        for _ in range(4 if goal_for_duration is not None else 1):
            goal = GoalToolPose.from_poses({"panda_hand": hand_pose})
            plan = _plan_nearest_branch(planner, goal, current)
            if (goal_for_duration is None or plan is None or plan.success is None
                    or not plan.success.all().item()):
                break
            steps = int(plan.interpolated_last_tstep.max().item())
            arrival = goal_for_duration(
                steps * env.step_dt * repeat_factor + settle_seconds)
            displacement = torch.linalg.vector_norm(
                arrival.position - hand_pose.position, dim=-1).max().item()
            hand_pose = arrival
            if displacement <= 0.001:
                break
    finally:
        if allow_gripper_contact:
            planner.enable_link_collision(contact_links)
        if linear_insertion:
            planner.update_tool_pose_criteria({"panda_hand": ToolPoseCriteria()})
    if plan is None or plan.success is None or not plan.success.all().item():
        status = "no plan" if plan is None else getattr(plan, "status", plan.success)
        world_goal = Pose(robot.data.root_pos_w, robot.data.root_quat_w).multiply(hand_pose)
        current_tcp = env.scene["ee_frame"].data.target_pos_w[:, 0]
        reachability = "not checked"
        try:
            ik = planner.ik_solver.solve_pose(goal, current)
            reachability = str(bool(ik.success.any().item()))
        except Exception as exc:
            reachability = f"check failed: {exc}"
        raise PlanningFailed(
            f"{label}: cuRobo found no collision-free arm path ({status}); "
            f"IK reachable={reachability}, current_tcp={current_tcp[0].tolist()}, "
            f"goal_hand_world={world_goal.position[0].tolist()}"
        )
    print(f"[PLAN] {label}", flush=True)
    execute_trajectory(env, app, plan, None, action, settle_seconds=settle_seconds,
                       repeat_factor=repeat_factor,
                       **({"interpolate_repeats": True} if interpolate_repeats else {}),
                       tracking_wait_s=getattr(env, "motion_tracking_wait_s", 0.0),
                       max_tracking_error=(getattr(env, "motion_insertion_tracking_tolerance_rad",
                                                  getattr(env, "motion_tracking_tolerance_rad", 0.10))
                                           if linear_insertion else
                                           0.08 if label == "dynamic_source_grasp" else
                                           0.06 if label in ("dynamic_transfer",
                                                             "dynamic_transfer_correction")
                                           else getattr(env, "motion_tracking_tolerance_rad", 0.05)))


def _move(env, app, planner, action, position, label, *, quaternion=None,
          allow_gripper_contact=False):
    from benchmark.backends.gsmini.workspace import grasp_poses

    hand_pose = grasp_poses(env, position, quaternion=quaternion)
    _plan_pose(env, app, planner, action, hand_pose, label,
               allow_gripper_contact=allow_gripper_contact)


def _rotation_pose(quaternion):
    """Create a cuRobo pose for quaternion-only operations."""
    import torch
    from curobo.types import Pose

    return Pose(torch.zeros_like(quaternion[..., :3]), quaternion)


def _tcp_object_pose(env):
    from curobo.types import Pose

    tcp = env.scene["ee_frame"].data
    obj = env.scene["object"].data
    return Pose(tcp.target_pos_w[:, 0], tcp.target_quat_w[:, 0]).inverse().multiply(
        Pose(obj.root_pos_w, obj.root_quat_w))


def _gripper_opening(env):
    geometry = getattr(env.cfg, "grasp_geometry", None)
    if geometry is not None:
        return geometry(env)[0]
    robot = env.scene["robot"]
    fingers, _ = robot.find_joints(env.cfg.actions.gripper_action.joint_names)
    return (robot.data.joint_pos[:, fingers].sum(dim=-1)
            + getattr(env.cfg, "grasp_width_offset", 0.0))


def _open_gripper(env, app, action, max_seconds=0.5, *, require_fully_open=True):
    """Observe two open samples; final release may use its original time budget."""
    import math
    from benchmark.backends.gsmini.execution import step_env

    robot = env.scene["robot"]
    ids, _ = robot.find_joints(env.cfg.actions.gripper_action.joint_names)
    target = env.action_manager.get_term("gripper_action")._open_command
    action[:, -1] = 1.0
    quiet = 0
    for _ in range(math.floor(max_seconds / env.step_dt)):
        step_env(env, app, action)
        error = (robot.data.joint_pos[:, ids] - target).abs().max().item()
        if not math.isfinite(error):
            raise RuntimeError("Nonfinite gripper state during opening")
        quiet = quiet + 1 if error <= 0.001 else 0
        if quiet >= 2:
            return
    if require_fully_open:
        raise RuntimeError("The gripper did not fully open before retreat")


def _verify_pose_stability(env, app, action, *, context: str,
                           max_position_drift: float = 0.005,
                           max_orientation_drift: float = 0.10,
                           max_wait_seconds: float | None = None,
                           pose_reader=None, advance=None):
    """Check a 0.5-second pose window using the caller's frame and step when supplied."""
    import math
    from curobo.types import Pose
    from benchmark.backends.gsmini.execution import step_env
    from benchmark.tasks.success import PoseStabilityWindow

    data = env.scene["object"].data
    if pose_reader is None:
        pose_reader = lambda: Pose(data.root_pos_w, data.root_quat_w)
    if advance is None:
        advance = lambda: step_env(env, app, action)

    window = PoseStabilityWindow(
        pose_reader(), step_dt=env.step_dt, max_position_drift=max_position_drift,
        max_orientation_drift=max_orientation_drift)
    if max_wait_seconds is not None and (
            not math.isfinite(max_wait_seconds) or max_wait_seconds < 0.5):
        raise ValueError("max_wait_seconds must be finite and at least 0.5 seconds")
    limit = window.steps if max_wait_seconds is None else math.ceil(max_wait_seconds / env.step_dt)
    for _ in range(limit):
        advance()
        outside, position_drift, orientation_drift = window.observe(pose_reader())
        if outside and max_wait_seconds is None:
            raise RuntimeError(
                f"{context} exceeded pose drift limits: "
                f"position drift={position_drift:.4f}m, "
                f"orientation drift={orientation_drift:.3f}rad")
        if window.ready:
            print(f"[STABLE] {context}: {window.steps * env.step_dt:.2f}s, "
                  f"peak drift={window.peak_position_drift:.4f}m, "
                  f"{window.peak_orientation_drift:.3f}rad", flush=True)
            return window.peak_position_drift, window.peak_orientation_drift
    raise RuntimeError(f"{context} did not stabilize within {limit * env.step_dt:g}s")


def _check_grasp_motion(env):
    import torch

    expected = env.grasp_reference
    actual = _tcp_object_pose(env)
    opening = _gripper_opening(env)
    center_local = getattr(env, "grasp_center_local", None)
    if center_local is None:
        position_error = torch.linalg.vector_norm(actual.position - expected.position, dim=-1)
    else:
        center_local = center_local.expand_as(expected.position)
        position_error = torch.linalg.vector_norm(
            actual.transform_points(center_local) - expected.transform_points(center_local),
            dim=-1,
        )
    angle_error = actual.angular_distance(expected)
    # Center tracking allows rolling payloads to rotate. Threaded payloads
    # must also retain their grasp orientation to transmit screw torque.
    orientation_required = center_local is None or getattr(env, "require_grasp_orientation", False)
    orientation_valid = ((angle_error <= 0.35).all() if orientation_required else
                         (torch.isfinite(angle_error).all()
                          & (torch.linalg.vector_norm(actual.quaternion, dim=-1) > 0).all()))
    valid = (torch.isfinite(torch.cat((actual.position, actual.quaternion), dim=-1)).all()
             & (position_error <= 0.02).all() & orientation_valid
             & torch.isfinite(opening).all() & (opening > 0.001).all())
    if not valid.item():
        robot = env.scene["robot"]
        finger_ids, _ = robot.find_joints(env.cfg.actions.gripper_action.joint_names)
        tcp_position = env.scene["ee_frame"].data.target_pos_w[0, 0]
        object_position = env.scene["object"].data.root_pos_w[0]
        raise RuntimeError(
            "Grasp lost: object moved relative to the TCP "
            f"(position={position_error.max().item():.4f}m, "
            f"xyz={(actual.position - expected.position)[0].tolist()}, "
            f"angle={angle_error.max().item():.3f}rad, "
            f"sim_time={env.common_step_counter * env.step_dt:.2f}s, "
            f"tcp_world={tcp_position.tolist()}, object_world={object_position.tolist()}, "
            f"finger_q={robot.data.joint_pos[0, finger_ids].tolist()}, "
            f"opening={opening.max().item():.4f}m, "
            f"object_velocity={env.scene['object'].data.root_vel_w[0].tolist()})")


def _object_pose_hand_goal(env, desired, *, grasp=None):
    """Convert a desired grasped-object pose to cuRobo's panda_hand frame."""
    from benchmark.backends.gsmini.workspace import grasp_poses

    tcp_to_object = _tcp_object_pose(env) if grasp is None else grasp
    tcp_goal = desired.multiply(tcp_to_object.inverse())
    return grasp_poses(env, tcp_goal.position, quaternion=tcp_goal.quaternion)


def _move_object_pose(env, app, planner, action, desired, label, *, repeat_factor=1,
                      goal_for_duration=None, settle_seconds=None,
                      allow_gripper_contact=True):
    """Plan a hand pose that preserves the measured rigid grasp transform."""
    hand_goal = _object_pose_hand_goal(env, desired)
    hand_goal_for_duration = (
        None if goal_for_duration is None else
        lambda duration: _object_pose_hand_goal(env, goal_for_duration(duration))
    )
    _plan_pose(env, app, planner, action, hand_goal, label,
               allow_gripper_contact=allow_gripper_contact, repeat_factor=repeat_factor,
               goal_for_duration=hand_goal_for_duration,
               settle_seconds=settle_seconds)


def _make_pose_controller(env):
    """Create the shared absolute-pose DLS controller for local contact motions."""
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

    return DifferentialIKController(
        DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        num_envs=env.num_envs, device=env.device,
    )


def _servo_hand_pose(env, app, controller, action, hand_goal, *, center_joints=False, max_joint_step=0.08,
                     joint_margin=0.0):
    """Execute one bounded local IK step using the live PhysX hand Jacobian."""
    from benchmark.backends.gsmini.execution import step_env

    arm = env.action_manager.get_term("arm_action")
    action[:, :arm.action_dim] = arm.position_actions(_hand_pose_joint_target(
        env, controller, hand_goal, center_joints=center_joints,
        max_joint_step=max_joint_step, joint_margin=joint_margin))
    step_env(env, app, action)


def _hand_pose_joint_target(env, controller, hand_goal, *, center_joints=False, max_joint_step=0.08,
                            joint_margin=0.0):
    """Return arm joint targets for one bounded local IK step using the live PhysX hand Jacobian."""
    import torch
    from curobo.types import Pose
    from isaaclab.utils.math import matrix_from_quat, quat_inv

    robot = env.scene["robot"]
    arm = env.action_manager.get_term("arm_action")
    binding = getattr(env, "_pose_servo_binding", None)
    if binding is None or binding[0] is not robot or binding[1] is not arm:
        joint_ids, _ = robot.find_joints(arm.cfg.joint_names, preserve_order=arm.cfg.preserve_order)
        hand_id = robot.find_bodies("panda_hand")[0][0]
        body_index = hand_id - 1 if robot.is_fixed_base else hand_id
        columns = joint_ids if robot.is_fixed_base else [index + 6 for index in joint_ids]
        binding = env._pose_servo_binding = (robot, arm, joint_ids, hand_id, body_index, columns)
    _, _, joint_ids, hand_id, body_index, columns = binding
    jacobian = robot.root_physx_view.get_jacobians()[:, body_index, :, columns].clone()
    rotation = matrix_from_quat(quat_inv(robot.data.root_quat_w))
    jacobian[:, :3] = rotation @ jacobian[:, :3]
    jacobian[:, 3:] = rotation @ jacobian[:, 3:]
    base = Pose(robot.data.root_pos_w, robot.data.root_quat_w)
    actual = base.inverse().multiply(Pose(
        robot.data.body_pos_w[:, hand_id], robot.data.body_quat_w[:, hand_id]))
    current = robot.data.joint_pos[:, joint_ids]
    controller.set_command(torch.cat((hand_goal.position, hand_goal.quaternion), dim=-1))
    delta = controller.compute(actual.position, actual.quaternion, jacobian, current) - current
    limits = robot.data.soft_joint_pos_limits[:, joint_ids]
    if joint_margin:
        limits = limits + limits.new_tensor([joint_margin, -joint_margin])
    if center_joints:
        projector = torch.eye(len(joint_ids), device=env.device)[None] - torch.linalg.pinv(jacobian) @ jacobian
        centering = (limits[:, :, 0] + limits[:, :, 1]) / 2 - current
        null_step = 0.05 * (projector @ centering[:, :, None]).squeeze(-1)
        primary = current + delta
        room = torch.where(null_step > 0, limits[:, :, 1] - primary,
                           primary - limits[:, :, 0])
        fraction = torch.where(null_step.abs() > 1e-12,
                               room / null_step.abs().clamp_min(1e-12), torch.ones_like(null_step))
        scale = fraction.amin(dim=-1, keepdim=True).clamp(0.0, 1.0)
        inside = ((primary >= limits[:, :, 0]) & (primary <= limits[:, :, 1])).all(dim=-1, keepdim=True)
        # Scale the whole nullspace step: clipping individual joints moves the TCP.
        delta += torch.where(inside, scale, 0.0) * null_step
    if not torch.isfinite(delta).all():
        raise RuntimeError("Local IK produced a non-finite joint command")
    delta *= max_joint_step / delta.abs().amax(dim=-1, keepdim=True).clamp_min(max_joint_step)
    return torch.maximum(torch.minimum(current + delta, limits[:, :, 1]), limits[:, :, 0])
