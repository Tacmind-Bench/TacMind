"""TacDynamic expert episode collection and motion policy."""

from __future__ import annotations

from collections import deque
import math

from benchmark.tasks.success import _dynamic_success, verify_task_success
from benchmark.tasks.dynamic.scene import TASK_IDS, resolve_gsmini_task, dynamic_goal_quaternion, dynamic_asset_scale, dynamic_payload_task, _belt_range, configure_conveyor_motor, sample_dynamic_reset, _verify_source_seating, _prepare_dynamic_scene, _dynamic_obstacle_paths, _make_env_cfg, _load_dynamic_stage


def _source_grasp_velocity(env, task):
    """Predict source motion along the carrier, without contact velocity jitter."""
    import torch

    velocity = torch.zeros_like(env.scene["object"].data.root_pos_w)
    if task["episode_config"]["profile"] == "platform_to_table":
        velocity[:, 1] = env.scene["platform"].data.root_vel_w[:, 1]
    return velocity


def _source_key_grasp_quaternion(source_quaternion):
    """Pinch the flat faces of a key that can rotate freely in its source lock."""
    from isaaclab.utils.math import quat_mul

    local = source_quaternion.new_tensor((0.0, 0.0, 1.0, 0.0))
    return quat_mul(source_quaternion, local.expand_as(source_quaternion))


def _close_moving_source_key(env, app, controller, action, goal):
    """Establish a bounded physical pinch before extracting the free key."""
    import torch
    from benchmark.contracts import gripper_action_metadata
    from benchmark.backends.gsmini.motion import _servo_hand_pose

    robot = env.scene["robot"]
    fingers, _ = robot.find_joints(env.cfg.actions.gripper_action.joint_names)
    mapping = gripper_action_metadata(env)
    opened = action.new_tensor(mapping["open"]).mean()
    closed = action.new_tensor(mapping["closed"]).mean()
    pending = robot.data.joint_pos[:, fingers].mean(dim=-1).clone()
    quiet = 0
    touched = False
    for index in range(math.ceil(10.0 / env.step_dt)):
        measured = robot.data.joint_pos[:, fingers].mean(dim=-1)
        if not quiet:
            # Traverse the open gap faster, then retain the accepted contact
            # increments. Any measured contact latches the fine approach.
            airborne = not touched and (pending > closed + 0.006).all()
            # Once contact blocks finger travel, allow the small preload target
            # to advance without waiting for the fingers to pass through the key.
            increment, lead = (0.002, 0.003) if airborne else (0.00025, 0.001 if touched else 0.0005)
            floor = closed + 0.006 if airborne else closed
            candidate = (pending - increment).clamp_min(floor)
            pending = torch.where(measured - candidate <= lead, candidate, pending)
        action[:, -1] = (2 * (pending - closed) / (opened - closed) - 1).clamp(-1, 1)
        _servo_hand_pose(env, app, controller, action, goal())
        forces = torch.stack([
            env.scene[f"gsmini_{side}_contact"].data.force_matrix_w.sum(dim=(1, 2))
            for side in ("left", "right")], dim=1).norm(dim=-1)
        if not torch.isfinite(forces).all():
            raise RuntimeError("Moving key closure has nonfinite target contact forces")
        touched = touched or (forces > 0.0).any().item()
        preloaded = (pending <= closed).all() and (forces >= 0.1).all()
        quiet = quiet + 1 if preloaded else 0
        if index % 20 == 0 or quiet >= 3:
            print(f"[KEY_CLOSE] step={index + 1}, target={pending.tolist()}, "
                  f"measured={measured.tolist()}, forces={forces.tolist()}, stable={quiet}", flush=True)
        if quiet >= 3:
            return
    raise RuntimeError(
        "Moving key did not establish stable bilateral contact within 10s: "
        f"forces={forces.tolist()}, target={pending.tolist()}, "
        f"measured={robot.data.joint_pos[:, fingers].mean(dim=-1).tolist()}")


def _future_fixture_position(target, duration: float, velocity_y=None, *, position=None):
    position = (target.root_pos_w if position is None else position).clone()
    # The conveyor constrains carrier motion to Y; X/Z velocity is contact jitter.
    position[:, 1] += (target.root_vel_w[:, 1] if velocity_y is None else velocity_y) * duration
    return position


def _key_turn_delta(start_yaw, end_yaw, wrist, limits):
    """Choose the equivalent yaw path with more available physical wrist travel."""
    import torch

    positive = torch.remainder(end_yaw - start_yaw, 2 * math.pi)
    negative = positive - 2 * math.pi
    # After alignment the downward hand's seventh axis opposes holder-local Z.
    def margin(delta):
        return torch.minimum(wrist - delta - limits[:, 0], limits[:, 1] - wrist + delta)
    return torch.where(margin(positive) >= margin(negative), positive, negative)


def _extract_moving_source_key(env, app, action, task, lift_distance):
    """Recenter and turn the real key through its lock before axial extraction."""
    import torch
    from curobo.types import Pose
    from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz, quat_slerp
    from benchmark.backends.gsmini.motion import _make_pose_controller, _object_pose_hand_goal, _servo_hand_pose, _tcp_object_pose

    obj, holder = env.scene['object'].data, env.scene['source_holder'].data
    controller = _make_pose_controller(env)

    def actual_local():
        return Pose(holder.root_pos_w, holder.root_quat_w).inverse().multiply(
            Pose(obj.root_pos_w, obj.root_quat_w))

    def move(end, phase):
        start = actual_local().clone()
        grasp = _tcp_object_pose(env).clone()
        distance = (end.position - start.position).norm(dim=-1).max().item()
        angle = start.angular_distance(end).max().item()
        turn_delta = None
        if phase == "turn_into_source_opening":
            robot = env.scene["robot"]
            wrist = robot.find_joints("panda_joint7")[0][0]
            joint = robot.data.joint_pos[:, wrist]
            limits = robot.data.soft_joint_pos_limits[:, wrist]
            _, _, start_yaw = euler_xyz_from_quat(start.quaternion)
            _, _, end_yaw = euler_xyz_from_quat(end.quaternion)
            turn_delta = _key_turn_delta(start_yaw, end_yaw, joint, limits)
            angle = turn_delta.abs().max().item()
            print(f"[KEY_TURN] yaw={start_yaw.tolist()}, wrist={joint.tolist()}, "
                  f"delta={turn_delta.tolist()}", flush=True)
        pace = (3.0 if phase in ("turn_into_source_opening", "axial_extract")
                else 2.0 if phase in ("recenter_x", "slide_to_opposite_outlet") else 1.0)
        increment = min(1.0, pace * 0.0005 / max(distance, 1e-12),
                        pace * 0.025 / max(angle, 1e-12))
        progress, quiet = 0.0, 0
        def interpolated(fraction):
            quaternion = (quat_from_euler_xyz(torch.zeros_like(start_yaw), torch.zeros_like(start_yaw),
                                             start_yaw + turn_delta * fraction) if turn_delta is not None
                          else torch.stack([quat_slerp(a, b.clone(), fraction)
                                            for a, b in zip(start.quaternion, end.quaternion)]))
            return Pose(torch.lerp(start.position, end.position, fraction), quaternion)

        for index in range(math.ceil(8.0 / env.step_dt)):
            actual = actual_local()
            proposed = min(1.0, progress + increment)
            pending = interpolated(proposed)
            if ((pending.position - actual.position).norm(dim=-1).max() <= pace * 0.0015
                    and pending.angular_distance(actual).max() <= pace * 0.04):
                progress = proposed
            pending = interpolated(progress)
            # The original 0.1 s carrier feedforward plus half a held action frame.
            future = holder.root_pos_w.clone()
            future[:, 1] += env.scene['platform'].data.root_vel_w[:, 1] * (0.1 + env.step_dt / 2)
            desired = Pose(future, holder.root_quat_w).multiply(pending)
            # Retain the grasp frame while advancing the constrained path.
            # At its endpoint, correct measured pad slip instead of waiting
            # forever for the original rigid-grasp approximation to converge.
            hand_goal = (_object_pose_hand_goal(env, desired, grasp=grasp) if progress < 1.0
                         else _object_pose_hand_goal(env, desired))
            # Keep the measured pose lead above; brake before the rotation ends.
            turning = phase == 'turn_into_source_opening' and progress < 0.85
            _servo_hand_pose(env, app, controller, action, hand_goal,
                             center_joints=True, max_joint_step=0.16 if turning else 0.08)
            actual = actual_local()
            error = (actual.position - end.position).norm(dim=-1).max().item()
            angle_error = actual.angular_distance(end).max().item()
            # Lateral contact can rock the key in the long slot. Once centered,
            # the next phase straightens it in the round bore; forcing the old
            # tilt here loads the slot walls until the key slips out of the pads.
            angle_tolerance = 0.06 if phase == "recenter_x" else 0.004
            quiet = quiet + 1 if progress == 1.0 and error <= 0.00035 and angle_error <= angle_tolerance else 0
            if index % 20 == 0 or quiet >= 3:
                print(f'[KEY_SOURCE] phase={phase}, step={index + 1}, progress={progress:.4f}, '
                      f'position={actual.position[0].tolist()}, error={error:.6f}, '
                      f'angle_error={angle_error:.5f}, stable={quiet}', flush=True)
            if quiet >= 3:
                return
        raise RuntimeError(f'Moving key {phase} did not physically converge within 8s: '
                           f'position_error={error:.6f}m, angle_error={angle_error:.5f}rad')

    initial = actual_local().clone()
    nominal = initial.position.new_tensor([task['episode_config']['source_payload_offset_m']])
    # Full-mesh source sweep: clear the chamber roofs, keeping the measured tilt.
    seated = initial.clone()
    seated.position[:, 2] = 0.117693868
    move(seated, 'lower_in_chamber')
    centered = seated.clone()
    centered.position[:, 0] = nominal[:, 0]
    move(centered, 'recenter_x')
    _, _, yaw = euler_xyz_from_quat(initial.quaternion)
    upright = centered.clone()
    upright.position[:, 1] = nominal[:, 1]
    upright.quaternion = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    move(upright, 'align_in_round_bore')
    aligned = upright.clone()
    # The equivalent opposite outlet avoids exhausting the real wrist's lower limit.
    aligned.quaternion = initial.quaternion.new_tensor([[0.0, 0.0, 0.0, -1.0]])
    move(aligned, 'turn_into_source_opening')
    aligned.position[:, 0] = 0.011885566
    move(aligned, 'slide_to_opposite_outlet')
    extracted = aligned.clone()
    extracted.position[:, 2] = initial.position[:, 2] + lift_distance
    move(extracted, 'axial_extract')


def _dynamic_fixture_pose(env, task: dict, local_position, duration: float):
    import torch
    from curobo.types import Pose

    target = env.scene["fixture"].data
    local = Pose(
        torch.tensor([local_position], device=env.device,
                     dtype=target.root_pos_w.dtype),
        torch.tensor([dynamic_goal_quaternion(task)], device=env.device,
                     dtype=target.root_quat_w.dtype),
    )
    fixture = Pose(_future_fixture_position(
        target, duration, getattr(env, "fixture_velocity_y", None)), target.root_quat_w)
    return fixture.multiply(local)


def _dynamic_goal(env, task: dict, duration: float = 0.0):
    return _dynamic_fixture_pose(
        env, task, task["episode_config"]["goal_offset_in_fixture_m"], duration,
    )


def _dynamic_insertion_goal(env, task: dict, duration: float = 0.0):
    local = [position + bias for position, bias in zip(
        task["episode_config"]["goal_offset_in_fixture_m"],
        task["episode_config"].get("insert_command_bias_m", (0.0, 0.0, 0.0)))]
    return _dynamic_fixture_pose(env, task, local, duration)


def _dynamic_entry(env, task: dict, duration: float = 0.0, clearance: float = 0.0):
    local_position = list(task["episode_config"]["target_entry_local_m"])
    local_position[2] += clearance
    return _dynamic_fixture_pose(env, task, local_position, duration)


def _dynamic_approach(env, task: dict, duration: float = 0.0):
    """Return the GSmini approach in the moving fixture's local frame."""
    episode = task["episode_config"]
    clearance = max(0.08, float(episode.get("carry_clearance_m", 0.10)))
    if task["task_id"] == 0 and "target_entry_local_m" in episode:
        return _dynamic_entry(env, task, duration, clearance)
    axis = episode["bore_axis_local"]
    length = math.sqrt(sum(value * value for value in axis))
    position = [value + clearance * axis[index] / length
                for index, value in enumerate(episode["goal_offset_in_fixture_m"])]
    return _dynamic_fixture_pose(env, task, position, duration)


def _wait_for_dynamic_target(env, app, planner, action, task, max_seconds=30.0):
    """Wait physically for an in-hand target to enter the arm's IK workspace."""
    from curobo.types import GoalToolPose
    from benchmark.backends.gsmini.motion import _object_pose_hand_goal, _planner_joint_state
    from benchmark.backends.gsmini.execution import hold

    interval = max(1, round(0.25 / env.step_dt)) * env.step_dt
    checks = math.ceil(max_seconds / interval)
    robot = env.scene["robot"]
    for index in range(checks + 1):
        current = _planner_joint_state(robot, planner)
        hand = _object_pose_hand_goal(env, _dynamic_approach(env, task))
        result = planner.ik_solver.solve_pose(
            GoalToolPose.from_poses({"panda_hand": hand}), current,
        )
        if bool(result.success.any().item()):
            if index:
                print(f"[WAIT] Moving target entered IK workspace after "
                      f"{index * interval:.2f}s", flush=True)
            return
        if index < checks:
            hold(env, app, action, interval)
    raise RuntimeError(f"Moving target did not enter IK workspace within {max_seconds:g}s")


def moving_insert_position(start_object, start_fixture, future_fixture, future_goal,
                           fraction):
    """Interpolate insertion depth in the frame of the moving fixture."""
    start_relative = start_object - start_fixture
    goal_relative = future_goal - future_fixture
    return future_fixture + start_relative + (goal_relative - start_relative) * fraction


def _insertion_fractions(env, task, goal, bore_world, travel, segment_length):
    """Coalesce airborne waypoints; keep the original waypoints near contact."""
    from itertools import product
    import torch
    from curobo.types import Pose
    from benchmark.backends.gsmini.environment import _asset_geometry

    episode = task["episode_config"]
    fixture = env.scene["fixture"].data
    projections = []
    for role, position, quaternion in (("fixture", fixture.root_pos_w, fixture.root_quat_w),
                                       ("object", goal.position, goal.quaternion)):
        asset_type = episode["fixture_asset_type" if role == "fixture" else "payload_asset_type"]
        low, high, _ = _asset_geometry(asset_type)
        corners = torch.tensor(list(product(*zip(low, high))), device=env.device,
                               dtype=position.dtype)
        corners *= torch.tensor(dynamic_asset_scale(task, role), device=env.device)
        pose = Pose(position - goal.position, quaternion)
        world = pose.batch_transform_points(corners[None].expand(position.shape[0], -1, -1).contiguous())
        projections.append((world * bore_world[:, None]).sum(dim=-1))
    # Bounding boxes are conservative: the tip stays 5 mm above the whole fixture.
    contact_travel = (projections[0].amax(dim=-1) - projections[1].amin(dim=-1)).clamp_min(0)
    contact_travel = contact_travel.max().item() + 0.005
    segments = max(1, math.ceil(travel / segment_length))
    first = max(1, math.floor(max(0.0, travel - contact_travel) / travel * segments)) if travel else 1
    # Keep moving-target updates frequent; static key and gear fixtures use longer spans.
    airborne_step = (0.04 if episode.get("object_family") in ("key", "gear")
                     and episode.get("profile") == "platform_to_table" else 0.02)
    stride = max(1, math.floor(airborne_step / segment_length))
    return [index / segments for index in (*range(stride, first, stride),
                                           *range(first, segments + 1))]


def _execute_insertion_fractions(fractions, travel, segment_length, move):
    """Split an unplannable airborne move before any physics steps execute."""
    from benchmark.backends.gsmini.environment import PlanningFailed

    pending = deque(fractions)
    previous = 0.0
    while pending:
        fraction = pending.popleft()
        try:
            move(fraction)
        except PlanningFailed:
            if (fraction - previous) * travel <= segment_length * (1 + 1e-6):
                raise
            midpoint = (previous + fraction) / 2
            pending.extendleft((fraction, midpoint))
            print(f"[INSERT_PLAN] Splitting airborne move at progress={midpoint:.3f}", flush=True)
        else:
            previous = fraction


def _bulb_support_errors(actual, supported):
    import torch
    from benchmark.backends.gsmini.motion import _rotation_pose

    axis = torch.zeros_like(actual.position)
    axis[:, 2] = 1.0
    actual_axis = _rotation_pose(actual.quaternion).transform_points(axis)
    supported_axis = _rotation_pose(supported.quaternion).transform_points(axis)
    tilt = torch.acos((actual_axis * supported_axis).sum(dim=-1).clamp(-1.0, 1.0))
    drift = torch.linalg.vector_norm((actual.position - supported.position)[:, :2], dim=-1)
    return drift.max().item(), tilt.max().item()


def _regrasp_threaded_bulb(env, app, action, future_goal, rewind_angle, *, axial_floor=-0.002,
                          tilt_limit=0.15, radial_limit=0.003,
                          lift_speed=0.05, rewind_speed=1.0, max_joint_step=0.2,
                          joint_margin=0.0, reuse_stability=False, gripper_speed=None,
                          use_wrist_travel=False, rewind_max_joint_step=None):
    """Open beyond the bulb diameter, rewind while tracking the socket and regrasp."""
    from curobo.types import Pose
    from benchmark.backends.gsmini.environment import _asset_geometry
    from benchmark.backends.gsmini.motion import _make_pose_controller, _rotation_pose, _servo_hand_pose, _tcp_object_pose, _verify_pose_stability
    from benchmark.expert.tabletop import _geometric_grasp_position, _threaded_object_quaternion
    from benchmark.backends.gsmini.workspace import grasp_poses
    from benchmark.tasks.success import PoseStabilityWindow

    data = env.scene["object"].data
    reference = future_goal(0.0)
    supported = reference.inverse().multiply(Pose(data.root_pos_w.clone(), data.root_quat_w.clone()))
    tcp = env.scene["ee_frame"].data
    local_tcp = reference.inverse().multiply(Pose(tcp.target_pos_w[:, 0].clone(),
                                                 tcp.target_quat_w[:, 0].clone()))
    controller = _make_pose_controller(env)
    stability = (PoseStabilityWindow(supported, step_dt=env.step_dt,
                                    max_position_drift=0.001, max_orientation_drift=0.02)
                 if reuse_stability else None)

    def supported_pose():
        return future_goal(0.0).inverse().multiply(Pose(data.root_pos_w, data.root_quat_w))

    def track(angle, *, follow_body=False, follow_axis=False, lift=0.0, joint_step=None):
        if follow_body:
            local_tcp.position = supported_pose().position + _rotation_pose(
                supported.quaternion).transform_points(grasp_point)
        goal = future_goal(0.2)
        quaternion = _rotation_pose(_threaded_object_quaternion(
            local_tcp.quaternion.new_tensor([[1.0, 0.0, 0.0, 0.0]]), angle)).multiply(
                _rotation_pose(local_tcp.quaternion)).quaternion
        position = local_tcp.position.clone()
        if follow_axis:
            actual_body = supported_pose()
            position = actual_body.transform_points(grasp_point)
            quaternion = _rotation_pose(actual_body.quaternion).multiply(
                _rotation_pose(supported.quaternion).inverse()).multiply(
                    _rotation_pose(quaternion)).quaternion
        position[:, 2] += lift
        desired = goal.multiply(Pose(position, quaternion))
        _servo_hand_pose(env, app, controller, action,
                         grasp_poses(env, desired.position, quaternion=desired.quaternion),
                         max_joint_step=max_joint_step if joint_step is None else joint_step,
                         joint_margin=joint_margin)
        actual = supported_pose()
        drift, tilt = _bulb_support_errors(actual, supported)
        drift = max(drift, actual.position[:, :2].norm(dim=-1).max().item())
        relative = supported.inverse().multiply(actual).quaternion[0]
        twist = 2 * math.atan2(abs(relative[3].item()), abs(relative[0].item()))
        height = actual.position[:, 2].min().item()
        lift = (actual.position[:, 2] - supported.position[:, 2]).max().item()
        if (not math.isfinite(drift) or not math.isfinite(tilt) or drift > radial_limit or tilt > tilt_limit
                or not math.isfinite(twist) or twist > 0.25
                or not math.isfinite(height) or height < axial_floor or lift > 0.003):
            raise RuntimeError(f"Bulb lost socket support during regrasp: radial={drift:.4f}m, axial={height:.4f}m, "
                               f"tilt={tilt:.3f}rad, twist={twist:.3f}rad, rewind={angle:.3f}rad")
        if stability is not None:
            stability.observe(actual)
        return desired

    robot = env.scene["robot"]
    _verify_pose_stability(
        env, app, action, context="Thread support before opening",
        max_position_drift=0.001, max_orientation_drift=0.02, max_wait_seconds=3.0,
        pose_reader=supported_pose, advance=lambda: track(0.0))
    supported = supported_pose().clone()
    tcp = env.scene["ee_frame"].data
    local_tcp = future_goal(0.0).inverse().multiply(Pose(
        tcp.target_pos_w[:, 0].clone(), tcp.target_quat_w[:, 0].clone()))
    if "grasp_point_local" in env.task["reset_contract"]:
        grasp_point = Pose(data.root_pos_w, data.root_quat_w).inverse().transform_points(
            _geometric_grasp_position(env, env.task))
    else:
        grasp_point = supported.inverse().transform_points(local_tcp.position).clone()
        grasp_point[:, :2] = 0.0
    _, maximum, _ = _asset_geometry("tactile_bulb")
    clearance = max(0.02, float(maximum[2]) + 0.02 - grasp_point[:, 2].min().item())
    env.grasp_reference = None
    finger_ids, _ = robot.find_joints(env.cfg.actions.gripper_action.joint_names)
    open_target = env.action_manager.get_term("gripper_action")._open_command
    initial_command = action[:, -1].clone()
    opening_seconds = 0.5 if gripper_speed is None else (1.0 - initial_command).abs().max().item() / gripper_speed
    opening_steps = max(1, math.ceil(opening_seconds / env.step_dt))
    for index in range(1, opening_steps + 1):
        action[:, -1] = initial_command + (1.0 - initial_command) * index / opening_steps
        track(0.0)
    action[:, -1] = 1.0
    for _ in range(25):
        if (robot.data.joint_pos[:, finger_ids] - open_target).abs().max().item() < 0.001:
            break
        track(0.0)
    else:
        raise RuntimeError("Bulb gripper did not fully open before regrasp movement")
    clearance_steps = max(1, math.ceil(clearance / (lift_speed * env.step_dt)))
    for index in range(1, clearance_steps + 1):
        track(0.0, follow_body=True, lift=clearance * index / clearance_steps)
    for _ in range(25):
        desired = track(0.0, follow_body=True, lift=clearance)
        actual_z = env.scene["ee_frame"].data.target_pos_w[:, 0, 2]
        if (actual_z >= desired.position[:, 2] - 0.001).all():
            break
    else:
        raise RuntimeError("Bulb gripper did not reach clearance before wrist rewind")
    if use_wrist_travel:
        # Regrasp at a new azimuth on the round bulb to use the remaining wrist
        # travel. Keep a margin to its existing limit and the clearance checks.
        wrist, = robot.find_joints("panda_joint7")[0]
        available = (robot.data.joint_pos[:, wrist]
                     - robot.data.soft_joint_pos_limits[:, wrist, 0] - 0.3).min().item()
        rewind_angle += max(0.0, min(0.8, available - rewind_angle))
    rewind_steps = math.ceil(abs(rewind_angle) / (rewind_speed * env.step_dt))
    # The larger target lead is confined to the open gripper above the bulb.
    for index in range(1, rewind_steps + 1):
        track(rewind_angle * index / rewind_steps, follow_axis=True, lift=clearance,
              joint_step=rewind_max_joint_step)
    for _ in range(25):
        desired = track(rewind_angle, follow_axis=True, lift=clearance,
                        joint_step=rewind_max_joint_step)
        tcp = env.scene["ee_frame"].data
        actual = Pose(tcp.target_pos_w[:, 0], tcp.target_quat_w[:, 0])
        if actual.angular_distance(desired).max().item() < 0.02:
            break
    else:
        raise RuntimeError(f"Bulb regrasp wrist did not finish rewinding before closure: "
                           f"angle={actual.angular_distance(desired).max().item():.4f}rad, "
                           f"joints={env.scene['robot'].data.joint_pos.tolist()}")
    for index in range(1, clearance_steps + 1):
        track(rewind_angle, follow_axis=True, lift=clearance * (1 - index / clearance_steps))
    close_command = float(env.task["reset_contract"].get("preclose_command", -1.0))
    for _ in range(25):
        desired = track(rewind_angle, follow_axis=True)
        tcp = env.scene["ee_frame"].data
        actual = Pose(tcp.target_pos_w[:, 0], tcp.target_quat_w[:, 0])
        center = future_goal(0.0).transform_points(supported_pose().transform_points(grasp_point))
        error = (actual.position - center).norm(dim=-1).max().item()
        if error < 0.001 and actual.angular_distance(desired).max().item() < 0.02:
            break
    else:
        raise RuntimeError(f"Bulb regrasp did not reach the grasp center before closure: {error:.4f}m")
    closing_seconds = 0.3 if gripper_speed is None else abs(close_command - 1.0) / gripper_speed
    pressure_steps = max(1, math.ceil(closing_seconds / env.step_dt))
    for index in range(1, pressure_steps + 1):
        action[:, -1] = 1.0 + (close_command - 1.0) * index / pressure_steps
        track(rewind_angle, follow_axis=True)
    # Hold the closed grasp in the socket frame instead of chasing contact deflection.
    tcp = env.scene["ee_frame"].data
    local_tcp = future_goal(0.0).inverse().multiply(Pose(
        tcp.target_pos_w[:, 0].clone(), tcp.target_quat_w[:, 0].clone()))
    if stability is None or not stability.ready:
        _verify_pose_stability(
            env, app, action, context="Thread grasp after closing",
            max_position_drift=0.001, max_orientation_drift=0.02, max_wait_seconds=3.0,
            pose_reader=supported_pose, advance=lambda: track(0.0))
    env.grasp_reference = _tcp_object_pose(env)
    print("[THREAD] physical regrasp complete", flush=True)


def run(app, seed: int, output, task_id: int, *, session=None) -> None:
    if task_id not in TASK_IDS:
        raise ValueError(f"Unsupported TacDynamic task ID: {task_id}")

    from contextlib import ExitStack
    from functools import partial
    from pathlib import Path
    import torch
    from curobo.types import Pose
    from isaaclab.envs import ManagerBasedRLEnv
    from benchmark.catalog import ASSETS, ROBOT_ASSET, get_task
    from benchmark.backends.gsmini.motion import _attach_payload_for_planning, _check_grasp_motion, _create_motion_planner, _make_pose_controller, _move, _move_object_pose, _plan_pose, _tcp_object_pose
    from benchmark.backends.gsmini.environment import initialize_task_environment, _hold_current_arm_action
    from benchmark.backends.gsmini.recording import _finish_episode, _prepare_output_path, _record_step_and_flush, _start_episode

    from benchmark.expert.tabletop import _execute_threaded_insertion, _geometric_grasp_position, _initialize_in_hand
    from benchmark.backends.gsmini.execution import configure_shutdown, hold
    from benchmark.backends.gsmini.workspace import grasp_poses, read_collision_world, update_collision_world
    from benchmark.backends.gsmini.sensors import planner_robot as gsmini_planner_robot

    source_task = get_task("tac_dynamic", task_id)
    task = resolve_gsmini_task(source_task)
    episode = task["episode_config"]
    reset = sample_dynamic_reset(task, seed)
    _prepare_output_path(Path(output))
    reused = session is not None and session.env is not None
    if reused:
        paths = session.env.dynamic_reset_context[2]
        stage = session.env.sim.stage
    else:
        stage = _load_dynamic_stage(app, task)
        platform_drop_clearance = float(episode.get("platform_drop_clearance_m", 0.01))
        paths = _prepare_dynamic_scene(stage, task, reset,
                                      drop_clearance_m=platform_drop_clearance)
        configure_conveyor_motor(stage, reset["belt_speed_mps"])
        cfg = _make_env_cfg(seed, task, stage, paths)
    in_hand = episode.get("in_hand_reset", {}).get("mode") == "in_hand"
    threaded_insertion = in_hand and episode["object_family"] == "bulb"
    synthetic_task = dynamic_payload_task(task)
    obstacle_paths = _dynamic_obstacle_paths(paths)

    class RecordingEnv(ManagerBasedRLEnv):
        episode = None
        grasp_reference = None
        episode_start_step = None
        require_grasp_orientation = threaded_insertion
        fixture_pose_history = None
        fixture_velocity_y = None

        def step(self, action):
            if self.episode is not None:
                elapsed = (self.common_step_counter - self.episode_start_step) * self.step_dt
                if elapsed >= float(episode["episode_timeout_s"]):
                    raise RuntimeError(
                        f"TacDynamic {task_id} exceeded its {episode['episode_timeout_s']}s timeout"
                    )
                _record_step_and_flush(self, action)
            result = super().step(action)
            if in_hand:
                if self.fixture_pose_history is None:
                    self.fixture_pose_history = deque(maxlen=math.ceil(0.5 / self.step_dt) + 1)
                self.fixture_pose_history.append((
                    self.common_step_counter,
                    self.scene["fixture"].data.root_pos_w[:, 1].clone(),
                ))
                if len(self.fixture_pose_history) > 1:
                    first_step, first_position = self.fixture_pose_history[0]
                    last_step, last_position = self.fixture_pose_history[-1]
                    self.fixture_velocity_y = (last_position - first_position) / (
                        (last_step - first_step) * self.step_dt)
            terminated, truncated = result[2], result[3]
            if (terminated | truncated).any():
                manager = self.termination_manager
                fired = [name for name, value in zip(
                    manager.active_terms, manager._term_dones[0].tolist(),
                ) if value]
                position = self.scene["object"].data.root_pos_w[0].tolist()
                raise RuntimeError(
                    f"TacDynamic {task_id} environment terminated; terms={fired}, "
                    f"object_position={position}, timeout={truncated.tolist()}"
                )
            if self.grasp_reference is not None:
                _check_grasp_motion(self)
            return result

    with ExitStack() as resources:
        if reused:
            env = session.env
            session.reset("dynamic", task_id, seed)
        else:
            env = RecordingEnv(cfg=cfg)
            env.task = synthetic_task
            lifetime = session.resources if session is not None else resources
            initialize_task_environment(lifetime, env, seed, configure_shutdown, task=task)
            lifetime.callback(env.sim.stop)
            if session is not None:
                from benchmark.tasks.dynamic.scene import bind_dynamic_reset
                bind_dynamic_reset(env, task, reset, paths)
                session.bind(env)
        action = torch.zeros_like(env.action_manager.action)
        _hold_current_arm_action(env, action)
        print(f"[RESET] TacDynamic {task_id}: object="
              f"{env.scene['object'].data.root_pos_w[0].tolist()}, fixture="
              f"{env.scene['fixture'].data.root_pos_w[0].tolist()}, platform="
              f"{env.scene['platform'].data.root_pos_w[0].tolist()}", flush=True)
        if in_hand:
            env.sim.forward()
            env.scene.update(0.0)
            _initialize_in_hand(env, synthetic_task)
            close_command = float(episode["in_hand_reset"].get("preclose_command", -0.7))
            action[:, -1] = (close_command if threaded_insertion
                             else -1.0)
            hold(env, app, action, 0.5)
            object_data = env.scene["object"].data
            object_pose = Pose(object_data.root_pos_w, object_data.root_quat_w)
            tcp_position = env.scene["ee_frame"].data.target_pos_w[:, 0]
            env.grasp_center_local = object_pose.inverse().transform_points(tcp_position)
            env.grasp_reference = _tcp_object_pose(env)

        in_hand_reset = episode.get("in_hand_reset", {})
        hold_command = float(in_hand_reset.get(
            "gelsight_transport_command",
            in_hand_reset.get("gelsight_hold_command",
                           in_hand_reset.get("preclose_command", -0.7)),
        ))
        gripper_command = (hold_command if threaded_insertion else
                           -1.0 if in_hand else 1.0)
        action[:, -1] = gripper_command
        from benchmark.tasks.dynamic.scene import settle_dynamic_scene
        settle_dynamic_scene(env, app, action)

        if paths["source_holder"]:
            poses = {role: (env.scene[role].data.root_pos_w[0].cpu().tolist(),
                            env.scene[role].data.root_quat_w[0].cpu().tolist())
                     for role in ("object", "source_holder")}
            source_seating = _verify_source_seating(env.sim.stage, paths, poses)
            print(f"[SOURCE_SEATING] {source_seating}", flush=True)

        moving_asset = (env.scene["fixture"] if paths["fixture"] == "/World/PreviewMovingAsset"
                        else env.scene["object"])
        platform = env.scene["platform"].data
        carried_speed = abs(moving_asset.data.root_vel_w[0, 1].item())
        expected_speed = reset["belt_speed_mps"]
        belt_top_z = float(_belt_range(env.sim.stage)[1][2])
        platform_bottom_z = (platform.root_pos_w[0, 2].item()
                             - float(episode["platform_size_m"][2]) / 2)
        platform_clearance = platform_bottom_z - belt_top_z
        if abs(platform_clearance) > 0.005:
            raise RuntimeError(
                f"The platform did not settle on the physical conveyor: "
                f"bottom={platform_bottom_z:.5f}m, belt={belt_top_z:.5f}m, "
                f"clearance={platform_clearance:.5f}m, "
                f"velocity={platform.root_vel_w[0].tolist()}"
            )
        if abs(platform.root_vel_w[0, 1].item()) < expected_speed * 0.35:
            raise RuntimeError(
                f"PhysX did not carry the platform at the configured belt speed: "
                f"platform={platform.root_vel_w[0, 1].item():.5f}m/s, "
                f"asset={moving_asset.data.root_vel_w[0, 1].item():.5f}m/s, "
                f"expected={expected_speed:.5f}m/s"
            )
        if carried_speed < expected_speed * 0.25:
            holder_state = None
            if paths["source_holder"]:
                holder = env.scene["source_holder"].data
                holder_state = {
                    "position": holder.root_pos_w[0].tolist(),
                    "velocity": holder.root_vel_w[0].tolist(),
                }
            raise RuntimeError(
                "The dynamic task asset is not moving with its platform: "
                f"path={paths['fixture'] if paths['fixture'] == '/World/PreviewMovingAsset' else paths['object']}, "
                f"asset_position={moving_asset.data.root_pos_w[0].tolist()}, "
                f"asset_velocity={moving_asset.data.root_vel_w[0].tolist()}, "
                f"platform_position={platform.root_pos_w[0].tolist()}, "
                f"platform_velocity={platform.root_vel_w[0].tolist()}, "
                f"source_holder={holder_state}, expected_speed={expected_speed:.5f}m/s"
            )

        def read_world(_env, names=("Table",)):
            paths_to_read = [obstacle_paths[name] for name in names]
            return read_collision_world(
                _env, obstacle_paths=tuple(paths_to_read),
                robot_link_path="/World/FrankaArm/panda_link0",
            )

        def all_obstacles(exclude=()):
            return tuple(name for name in obstacle_paths if name not in exclude)

        planner_robot = partial(gsmini_planner_robot, robot_usd_path=str(ASSETS / ROBOT_ASSET))
        planner_world = read_world(env, all_obstacles(("object", "fixture")))
        planner_cache_world = read_world(env, all_obstacles())
        planner_options = dict(
            robot_config=planner_robot,
            velocity_scale=(0.5 if task_id == 0 else
                            1.0 if task_id in (1, 6) else 2 ** -0.5),
            trajectory_knots=(None if task_id in (0, 1) else
                              6 if task_id in (5, 10) else 8),
        )
        # Keep a stable factory identity so unchanged geometry retains its graph.
        if session is not None:
            if not hasattr(session, "dynamic_planner_robot"):
                session.dynamic_planner_robot = planner_robot
            planner_options["robot_config"] = session.dynamic_planner_robot
            previous_planner = session.planner
            planner = session.acquire_planner(
                env, seed, planner_world, planner_cache_world, **planner_options)
            print(f"[SESSION] environment_reused={reused}, "
                  f"planner_reused={planner is previous_planner}", flush=True)
        else:
            planner_options["robot_config"] = planner_robot
            planner = resources.enter_context(_create_motion_planner(
                env, seed, planner_world, planner_cache_world, **planner_options))
            planner.warmup(enable_graph=True, num_warmup_iterations=1)
        if in_hand:
            _wait_for_dynamic_target(env, app, planner, action, task)
        action[:, -1] = gripper_command
        staged = _start_episode(resources, env, Path(output), "tac_dynamic", task_id, seed,
                                task=task,
                                extra_env_args={
                                    "belt_speed_mps": expected_speed,
                                    "reset": reset,
                                })
        env.episode_start_step = env.common_step_counter

        if not in_hand:
            source = env.scene["object"].data
            grasp_position = _geometric_grasp_position(env, synthetic_task)
            grasp_offset = float(episode.get("grasp_offset_m", 0.0))
            grasp_xy_offset = tuple(map(float, episode.get("grasp_xy_offset_m", (0.0, 0.0))))
            if (len(grasp_xy_offset) != 2
                    or not all(math.isfinite(value) for value in grasp_xy_offset)
                    or not math.isfinite(grasp_offset)):
                raise ValueError("Invalid dynamic grasp offset")
            if grasp_offset or any(grasp_xy_offset):
                local_offset = torch.tensor([[*grasp_xy_offset, 0.0]],
                                            device=env.device,
                                            dtype=source.root_pos_w.dtype).expand_as(
                                                source.root_pos_w)
                local_offset = local_offset * torch.tensor(
                    dynamic_asset_scale(task, "object"), device=env.device,
                    dtype=source.root_pos_w.dtype,
                )
                local_offset[:, 2] += grasp_offset
                grasp_position = Pose(source.root_pos_w, source.root_quat_w).transform_points(
                    local_offset) + (grasp_position - source.root_pos_w)
            grasp_quaternion = episode.get("source_grasp_quaternion_wxyz")
            measured_key_attitude = task_id == 6 and grasp_quaternion is None
            if grasp_quaternion is None:
                if task_id == 8:
                    grasp_quaternion = (0.0, 0.0, 1.0, 0.0)
                elif episode["object_family"] in {"key", "gear"}:
                    grasp_quaternion = (0.0, 2**-0.5, 2**-0.5, 0.0)
            else:
                grasp_quaternion = tuple(map(float, grasp_quaternion))
                norm = math.sqrt(sum(value * value for value in grasp_quaternion))
                if (len(grasp_quaternion) != 4
                        or not all(math.isfinite(value) for value in grasp_quaternion)
                        or not math.isclose(norm, 1.0, abs_tol=1.0e-3)):
                    raise ValueError("Invalid source grasp quaternion")
            if measured_key_attitude:
                grasp_quaternion = _source_key_grasp_quaternion(source.root_quat_w.clone())
            grasp_local = Pose(source.root_pos_w, source.root_quat_w).inverse().transform_points(
                grasp_position)
            grasp_goal = grasp_poses(env, grasp_position, quaternion=grasp_quaternion)
            velocity = _source_grasp_velocity(env, task)
            hand_world_goal = Pose(env.scene["robot"].data.root_pos_w,
                                   env.scene["robot"].data.root_quat_w).multiply(grasp_goal)
            print(f"[GRASP_TARGET] object={source.root_pos_w[0].tolist()}, "
                  f"grasp={grasp_position[0].tolist()}, "
                  f"velocity={velocity[0].tolist()}, "
                  f"hand={hand_world_goal.position[0].tolist()}", flush=True)

            def grasp_target(clearance, duration):
                point = grasp_position + velocity * duration
                point = point.clone()
                point[:, 2] += clearance
                return grasp_poses(env, point, quaternion=grasp_quaternion)

            update_collision_world(planner, read_world(env, all_obstacles(("object",))))
            _plan_pose(env, app, planner, action,
                       grasp_target(float(episode.get("transport_clearance_m", 0.10)), 0.0),
                       "dynamic_source_approach",
                       goal_for_duration=lambda duration: grasp_target(
                           float(episode.get("transport_clearance_m", 0.10)), duration),
                       settle_seconds=0.2,
                       repeat_factor=int(episode.get("source_approach_repeat_factor", 1)))
            grasp_position = Pose(source.root_pos_w, source.root_quat_w).transform_points(
                grasp_local)
            if measured_key_attitude:
                # Refresh after the unloaded approach, then retain this attitude
                # throughout descent and closure rather than chasing contact rocking.
                grasp_quaternion = _source_key_grasp_quaternion(source.root_quat_w.clone())
            velocity = _source_grasp_velocity(env, task)
            grasp_goal = grasp_poses(env, grasp_position, quaternion=grasp_quaternion)
            update_collision_world(planner, read_world(env, all_obstacles(("object",))))
            grasp_settle = float(episode.get(
                "source_grasp_settle_seconds", 0.2))
            _plan_pose(env, app, planner, action, grasp_goal, "dynamic_source_grasp",
                       allow_gripper_contact=True,
                       goal_for_duration=lambda duration: grasp_target(0.0, duration),
                       settle_seconds=grasp_settle)
            action[:, -1] = -1.0
            if episode["profile"] == "platform_to_table":
                from benchmark.backends.gsmini.motion import _servo_hand_pose
                close_controller = _make_pose_controller(env)
                if task_id == 6:
                    def key_close_goal():
                        position = Pose(source.root_pos_w, source.root_quat_w).transform_points(grasp_local)
                        position += _source_grasp_velocity(env, task) * 0.1
                        return grasp_poses(env, position, quaternion=grasp_quaternion)

                    _close_moving_source_key(env, app, close_controller, action, key_close_goal)
                else:
                    for _ in range(math.ceil(0.25 / env.step_dt)):
                        position = Pose(source.root_pos_w, source.root_quat_w).transform_points(grasp_local)
                        position += _source_grasp_velocity(env, task) * 0.1
                        _servo_hand_pose(env, app, close_controller, action,
                                         grasp_poses(env, position, quaternion=grasp_quaternion))
            else:
                hold(env, app, action, 0.25)
            print(f"[SOURCE_GRASP] object={source.root_pos_w[0].tolist()}, "
                  f"tcp={env.scene['ee_frame'].data.target_pos_w[0, 0].tolist()}, "
                  f"finger_q={env.scene['robot'].data.joint_pos[0, -2:].tolist()}, "
                  f"velocity={source.root_vel_w[0].tolist()}", flush=True)
            object_pose = Pose(source.root_pos_w, source.root_quat_w)
            tcp_position = env.scene["ee_frame"].data.target_pos_w[:, 0]
            env.grasp_center_local = object_pose.inverse().transform_points(tcp_position)
            env.grasp_reference = _tcp_object_pose(env)
            source_z = source.root_pos_w[:, 2].clone()
            lift_distance = max(0.05, float(episode.get("grasp_lock_lift_m", 0.0)))
            if task_id == 4:
                from benchmark.tasks.dynamic.scene import _seating_metrics

                seating = _seating_metrics(stage, paths, {
                    name: (env.scene[name].data.root_pos_w[0].tolist(),
                           env.scene[name].data.root_quat_w[0].tolist())
                    for name in ('object', 'source_holder')})
                # A nominal 5 cm lift barely reaches the source rim. Reserve
                # clearance for measured tracking error before tilting in transit.
                lift_distance = max(lift_distance, seating['insertion_depth_m'] + 0.01)
            lift_velocity_y = _source_grasp_velocity(env, task)[:, 1]
            lift_segment = float(episode.get("source_lift_segment_length_m", lift_distance))
            if not math.isfinite(lift_segment) or lift_segment <= 0:
                raise ValueError(f"Invalid source lift segment length: {lift_segment}")
            lift_segments = max(1, math.ceil(lift_distance / lift_segment))
            if task_id == 6:
                _extract_moving_source_key(env, app, action, task, lift_distance)
            else:
                for index in range(1, lift_segments + 1):
                    segment_start = source.root_pos_w.clone()
                    segment_height = source_z + lift_distance * index / lift_segments
                    segment_goal = segment_start.clone()
                    segment_goal[:, 2] = segment_height

                    def lift_goal(duration):
                        position = _future_fixture_position(
                            source, duration, velocity_y=lift_velocity_y, position=segment_goal)
                        return Pose(position, source.root_quat_w.clone())

                    _move_object_pose(env, app, planner, action,
                                      Pose(segment_goal, source.root_quat_w.clone()),
                                      "dynamic_grasp_lift", goal_for_duration=lift_goal,
                                      settle_seconds=0.05)
            if task_id in (4, 6):
                from benchmark.tasks.dynamic.scene import _seating_metrics

                seating = _seating_metrics(stage, paths, {
                    name: (env.scene[name].data.root_pos_w[0].tolist(),
                           env.scene[name].data.root_quat_w[0].tolist())
                    for name in ('object', 'source_holder')})
                if seating['insertion_depth_m'] > -0.003:
                    raise RuntimeError(f'Payload did not fully clear its source holder: {seating}')
                print(f'[SOURCE_CLEAR] {seating}', flush=True)
            lift_height = (source.root_pos_w[:, 2] - source_z).item()
            minimum_lift = float(episode.get("minimum_source_lift_m", 0.035))
            if not math.isfinite(minimum_lift) or minimum_lift <= 0:
                raise ValueError(f"Invalid minimum source lift: {minimum_lift}")
            if lift_height < minimum_lift:
                raise RuntimeError(
                    f"Payload did not physically lift from source: {lift_height:.4f}m "
                    f"(required {minimum_lift:.4f}m)"
                )

        target = env.scene["fixture"].data
        _attach_payload_for_planning(
            env, planner, all_obstacles(("object",)), read_world, "object",
            include_table=True, object_path=paths["object"],
        )
        def future_goal(duration):
            return _dynamic_goal(env, task, duration)

        transfer_repeat = int(episode.get(
            "transfer_repeat_factor", 1))
        transfer_settle = float(episode.get(
            "transfer_settle_seconds", 1.0 if threaded_insertion else 0.2))
        if not math.isfinite(transfer_settle) or transfer_settle < 0:
            raise ValueError(f"Invalid transfer settle time: {transfer_settle}")
        bore_local = torch.as_tensor(episode["bore_axis_local"], device=env.device,
                                     dtype=target.root_pos_w.dtype)
        bore_local /= torch.linalg.vector_norm(bore_local).clamp_min(1.0e-8)
        bore_world = Pose(torch.zeros_like(target.root_pos_w), target.root_quat_w).transform_points(
            bore_local.expand_as(target.root_pos_w))
        vertical_entry = task_id == 0 and "target_entry_local_m" in episode
        def approach_goal(duration):
            return _dynamic_approach(env, task, duration)

        transfer_tolerance = float(episode.get("transfer_approach_tolerance_m", 0.01))
        transfer_attempts = int(episode.get("transfer_correction_attempts", 2))
        if not math.isfinite(transfer_tolerance) or transfer_tolerance <= 0:
            raise ValueError(f"Invalid transfer approach tolerance: {transfer_tolerance}")
        if transfer_attempts < 0:
            raise ValueError(f"Invalid transfer correction attempts: {transfer_attempts}")
        approach = approach_goal(0.0)
        _move_object_pose(env, app, planner, action, approach, "dynamic_transfer",
                          allow_gripper_contact=False, goal_for_duration=approach_goal,
                          settle_seconds=transfer_settle,
                          repeat_factor=transfer_repeat)
        for _ in range(transfer_attempts):
            approach = approach_goal(0.0)
            transfer_error = torch.linalg.vector_norm(
                env.scene["object"].data.root_pos_w - approach.position, dim=-1,
            ).max().item()
            if transfer_error <= transfer_tolerance:
                break
            _move_object_pose(
                env, app, planner, action, approach, "dynamic_transfer_correction",
                allow_gripper_contact=False, goal_for_duration=approach_goal,
                settle_seconds=transfer_settle,
                repeat_factor=transfer_repeat,
            )
        approach = approach_goal(0.0)
        transfer_error = torch.linalg.vector_norm(
            env.scene["object"].data.root_pos_w - approach.position, dim=-1,
        ).max().item()
        if transfer_error > transfer_tolerance:
            actual_tcp = env.scene["ee_frame"].data.target_pos_w[:, 0]
            wanted_tcp = approach.multiply(env.grasp_reference.inverse()).position
            actual_grasp = _tcp_object_pose(env)
            grasp_center = getattr(env, "grasp_center_local", None)
            grasp_center_error = None
            if grasp_center is not None:
                grasp_center_error = torch.linalg.vector_norm(
                    actual_grasp.transform_points(grasp_center)
                    - env.grasp_reference.transform_points(grasp_center), dim=-1,
                ).max().item()
            print(f"[TRANSFER_DIAGNOSTIC] tcp_error="
                  f"{torch.linalg.vector_norm(actual_tcp - wanted_tcp, dim=-1).max().item():.4f}m, "
                  f"grasp_delta="
                  f"{(actual_grasp.position - env.grasp_reference.position)[0].tolist()}, "
                  f"grasp_angle={actual_grasp.angular_distance(env.grasp_reference).max().item():.4f}rad, "
                  f"grasp_center_error={grasp_center_error}, "
                  f"tcp={actual_tcp[0].tolist()}, wanted_tcp={wanted_tcp[0].tolist()}, "
                  f"object_velocity={env.scene['object'].data.root_vel_w[0].tolist()}", flush=True)
            raise RuntimeError(
                f"Dynamic transfer missed its approach pose by {transfer_error:.4f}m: "
                f"object={env.scene['object'].data.root_pos_w[0].tolist()}, "
                f"approach={approach.position[0].tolist()}, "
                f"fixture={target.root_pos_w[0].tolist()}, "
                f"fixture_velocity={target.root_vel_w[0, :3].tolist()}"
            )
        if vertical_entry:
            entry_goal = _dynamic_entry(env, task)
            _move_object_pose(
                env, app, planner, action, entry_goal, "dynamic_entry_align",
                goal_for_duration=lambda duration: _dynamic_entry(env, task, duration),
                settle_seconds=0.05,
            )

        update_collision_world(planner, read_world(env, all_obstacles(("object", "fixture"))))
        start = env.scene["object"].data.root_pos_w.clone()
        fixture_start = target.root_pos_w.clone()
        goal_now = _dynamic_goal(env, task)
        if threaded_insertion:
            goal_now.position += bore_world * float(episode["thread_engagement_range_m"][1])
        displacement = torch.dot((goal_now.position - start)[0], bore_world[0])
        travel = abs(displacement.item())
        insert_overshoot = float(episode.get("insert_overshoot_m", 0.0))
        if not math.isfinite(insert_overshoot) or insert_overshoot < 0:
            raise ValueError(f"Invalid insertion overshoot: {insert_overshoot}")
        if threaded_insertion:
            insert_overshoot = 0.0
        insert_settle = float(episode.get("insert_settle_seconds", 0.0))
        if not math.isfinite(insert_settle) or insert_settle < 0:
            raise ValueError(f"Invalid insertion settle time: {insert_settle}")
        insert_repeat = int(episode.get("insert_repeat_factor", 1))
        if threaded_insertion:
            insert_settle = max(0.2, insert_settle)
        if insert_repeat < 1:
            raise ValueError(f"Invalid insertion repeat factor: {insert_repeat}")
        travel += insert_overshoot
        segment_length = float(episode.get(
            "insert_segment_length_m", 0.005 if task_id == 0 else 0.01))
        if not math.isfinite(segment_length) or segment_length <= 0:
            raise ValueError(f"Invalid insertion segment length: {segment_length}")
        insertion_label = ("vertical_insertion" if vertical_entry else
                           "angled_insertion" if task_id == 0 else "dynamic_insert")
        segments = max(1, math.ceil(travel / segment_length))
        correction_limit = float(episode.get("insert_xy_correction_limit_m", 0.0))
        radial_tolerance = float(episode.get(
            "success_radial_tolerance_m", episode.get("success_xy_tolerance_m", 0.004)))
        if not math.isfinite(correction_limit) or correction_limit < 0:
            raise ValueError(f"Invalid insertion correction limit: {correction_limit}")
        if threaded_insertion:
            fractions = [1.0]
        elif task_id == 6:
            # The destination lock is stationary. Retain collision-checked
            # straight insertion and its subdivision fallback, with fewer stops.
            spans = max(1, math.ceil(travel / 0.02))
            fractions = [index / spans for index in range(1, spans + 1)]
        elif task_id not in (2, 10) and (in_hand or not episode.get("coalesce_airborne_insertions", False)):
            fractions = [index / segments for index in range(1, segments + 1)]
        else:
            fractions = _insertion_fractions(
                env, task, Pose(goal_now.position - bore_world * insert_overshoot,
                                goal_now.quaternion), bore_world, travel, segment_length)
        # The stationary angled tube bore needs continuous contact motion;
        # repeated short plans can rock the tube and unload its pinch.
        local_insert = task_id in (2, 4, 8)
        if local_insert:
            from benchmark.backends.gsmini.motion import _object_pose_hand_goal, _servo_hand_pose
            insert_controller = _make_pose_controller(env)
        previous_fraction = 0.0
        local_insert_done = False

        def insert_fraction(fraction):
            nonlocal previous_fraction, local_insert_done
            if local_insert_done:
                return
            def insert_goal(duration, fraction=fraction):
                final = (_dynamic_insertion_goal(env, task, duration) if local_insert
                         else future_goal(duration))
                if threaded_insertion:
                    final.position += bore_world * float(episode["thread_engagement_range_m"][1])
                future_fixture = _future_fixture_position(target, duration, env.fixture_velocity_y)
                position = moving_insert_position(
                    start, fixture_start, future_fixture,
                    final.position - bore_world * insert_overshoot, fraction)
                return Pose(position, final.quaternion)

            if local_insert:
                distance = (fraction - previous_fraction) * travel
                # Keep contact motion slower than coalesced airborne spans.
                speed = ((0.04 if task_id == 8 else 0.02)
                         if distance > segment_length * (1 + 1e-6)
                         else 0.01 if task_id == 4 else 0.005)
                steps = max(1, math.ceil(distance / (speed * env.step_dt)))
                advance_steps = steps
                if task_id == 2 and fraction == 1.0:
                    steps += math.ceil(5.0 / env.step_dt)
                quiet = 0
                for index in range(1, steps + 1):
                    current_fraction = previous_fraction + (fraction - previous_fraction) * min(index / advance_steps, 1.0)
                    if task_id == 2:
                        relative = Pose(target.root_pos_w, target.root_quat_w).inverse().multiply(
                            Pose(env.scene["object"].data.root_pos_w,
                                 env.scene["object"].data.root_quat_w))
                        height = relative.position[0, 2].item()
                        lag = ((env.scene["object"].data.root_pos_w
                                - insert_goal(0.0, current_fraction).position) * bore_world).sum().item()
                        if (lag > 0.001 and height
                                < episode["goal_offset_in_fixture_m"][2] + 0.006):
                            action[:, -1] = float(episode["transport_preload_command"])
                    desired = insert_goal(0.2, current_fraction)
                    _servo_hand_pose(env, app, insert_controller, action,
                                     _object_pose_hand_goal(env, desired))
                    quiet = quiet + 1 if _dynamic_success(env, task)["success"] else 0
                    if quiet >= int(episode.get("success_stable_steps", 3)):
                        if task_id in (4, 8):
                            _hold_current_arm_action(env, action)
                        local_insert_done = True
                        break
                previous_fraction = fraction
                delta = (env.scene["object"].data.root_pos_w - insert_goal(0.0).position)[0]
                print(f"[INSERT_TRACKING] fraction={fraction:.3f}, delta={delta.tolist()}, "
                      f"angle={Pose(quaternion=env.scene['object'].data.root_quat_w).angular_distance(insert_goal(0.0)).max().item():.3f}rad", flush=True)
            else:
                _move_object_pose(env, app, planner, action, insert_goal(0.0),
                                  insertion_label, repeat_factor=insert_repeat,
                                  goal_for_duration=insert_goal,
                                  settle_seconds=insert_settle)

        _execute_insertion_fractions(fractions, travel, segment_length, insert_fraction)

        if threaded_insertion:
            final = future_goal(0.0)
            engage = Pose(
                final.position + bore_world * float(episode["thread_engagement_range_m"][1]),
                final.quaternion,
            )
            success_steps = 0
            def threaded_success():
                nonlocal success_steps
                success_steps = success_steps + 1 if _dynamic_success(env, task)["success"] else 0
                return success_steps >= int(episode.get("success_stable_steps", 3))
            _execute_threaded_insertion(
                env, app, planner, action, engage, final.position, final.quaternion,
                # Anticipate the measured actuator lag while the socket moves.
                goal_for_duration=future_goal, tracking_horizon=0.2,
                angular_speed=2.5,
                contact_progress=True, follow_grasp_translation=False,
                grasp_axial_correction=0.0015,
                engaged_angular_speed=1.5, engaged_max_joint_step=0.3,
                regrasp=lambda angle: _regrasp_threaded_bulb(env, app, action, future_goal, angle,
                    lift_speed=0.3, rewind_speed=3.5, max_joint_step=0.3,
                    joint_margin=0.05, reuse_stability=True, gripper_speed=10.0,
                    use_wrist_travel=True,
                    axial_floor=float(episode["success_axial_range_m"][0]),
                    tilt_limit=math.acos(float(episode["success_axis_alignment"])),
                    radial_limit=float(episode["success_radial_tolerance_m"])),
                success_check=threaded_success,
            )

        if vertical_entry and not _dynamic_success(env, task)["success"]:
            _move_object_pose(
                env, app, planner, action, future_goal(0.0), "vertical_insertion",
                repeat_factor=insert_repeat, goal_for_duration=future_goal,
                settle_seconds=0.0,
            )
            print(f"[INSERT_CORRECTION] {_dynamic_success(env, task)}", flush=True)

        if correction_limit:
            object_pose = Pose(env.scene["object"].data.root_pos_w,
                               env.scene["object"].data.root_quat_w)
            fixture_pose = Pose(target.root_pos_w, target.root_quat_w)
            relative = fixture_pose.inverse().multiply(object_pose)
            error = relative.position[0] - torch.as_tensor(
                episode["goal_offset_in_fixture_m"], device=env.device,
                dtype=relative.position.dtype,
            )
            axial_error = torch.dot(error, bore_local)
            radial_error = error - axial_error * bore_local
            radial_distance = torch.linalg.vector_norm(radial_error).item()
            if radial_tolerance < radial_distance <= correction_limit:
                corrected_relative = Pose(relative.position - radial_error,
                                          relative.quaternion)

                def corrected_goal(duration):
                    future_fixture = Pose(
                        _future_fixture_position(target, duration, env.fixture_velocity_y),
                        target.root_quat_w,
                    )
                    return future_fixture.multiply(corrected_relative)

                _move_object_pose(env, app, planner, action, corrected_goal(0.0),
                                  "dynamic_realign",
                                  goal_for_duration=corrected_goal,
                                  settle_seconds=0.1)
                hold(env, app, action, 0.2)

        if in_hand and not threaded_insertion:
            # Short trajectories can stop above a contacted moving socket.
            # Recompute the grasp transform and target from measured PhysX
            # poses, instead of accepting the commanded endpoint as insertion.
            for _ in range(int(episode.get("max_fine_align", 6))):
                if _dynamic_success(env, task)["success"]:
                    break
                _move_object_pose(
                    env, app, planner, action, future_goal(0.0), "vertical_insertion",
                    repeat_factor=insert_repeat, goal_for_duration=future_goal,
                    settle_seconds=0.0,
                )
                print(f"[INSERT_CORRECTION] {_dynamic_success(env, task)}", flush=True)

        print(f"[INSERT_RESULT] {_dynamic_success(env, task)}", flush=True)
        stable_steps = int(episode.get("success_stable_steps", 3))
        if episode.get("release_after_insert", False):
            env.grasp_reference = None
            action[:, -1] = 1.0
            hold(env, app, action, 0.2)
            update_collision_world(planner, read_world(env, all_obstacles()))
            planner.attachment_manager.detach()
            retreat = env.scene["ee_frame"].data.target_pos_w[:, 0].clone()
            retreat[:, 2] += 0.08
            _move(env, app, planner, action, retreat, "dynamic_retreat",
                  allow_gripper_contact=True)
        else:
            hold(env, app, action, stable_steps * env.step_dt)

        result = verify_task_success(env, app, action, task, suite="tac_dynamic", backend="gsmini")
        if task_id == 8:
            print(f"[TARGET_SEATING] {result['target_insertion']}", flush=True)
        _finish_episode(env, Path(output), staged)
        print(f"[SUCCESS] TacDynamic {task_id}; belt={expected_speed:.4f}m/s, "
              f"radial={result['radial_error_m']:.4f}m, "
              f"axial={result['axial_error_m']:.4f}m, "
              f"axis={result['axis_alignment']:.3f}; saved {output}", flush=True)
