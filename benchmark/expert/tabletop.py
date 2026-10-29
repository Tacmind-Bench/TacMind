"""Shared tabletop expert strategies for Precision, Recog and Extra."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from functools import partial
from pathlib import Path

from benchmark.tasks.success import TaskSuccessMonitor, check_task_success, _task_object_goal, _insertion_errors, verify_task_success
from benchmark.catalog import TABLETOP_IDENTITIES, get_task, tabletop_variant
from benchmark.scene.reset import robot_joint_defaults
from benchmark.expert.capabilities import REUSABLE_GSMINI_TASK_IDS, SHARED_TABLETOP_EXPERT_FAMILIES
from benchmark.backends.gsmini.environment import GSminiCollectionSession, _asset_geometry, _hold_current_arm_action, initialize_task_environment, _make_env_cfg, reset_tabletop_task
from benchmark.backends.gsmini.motion import _attach_payload_for_planning, _check_grasp_motion, _create_motion_planner, _gripper_opening, _make_pose_controller, _move, _move_object_pose, _nearest_joint_solution, _object_pose_hand_goal, _open_gripper, _planner_joint_state, _rotation_pose, _tcp_object_pose, _verify_pose_stability
from benchmark.backends.gsmini.recording import _finish_episode, _prepare_output_path, _prepare_recording_step, _start_episode

BORE_RADIAL_CLEARANCE = 0.00035  # 0.30 mm nominal gap plus a 0.05 mm control tolerance.


class TubeCollectionSession:
    """Own one GSmini Precision 0 environment and warmed planner across recordings."""

    def __init__(self):
        self.resources = ExitStack()
        self.env = None
        self.planner = None
        self.planner_seed = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            if self.planner is not None:
                self.planner.destroy()
        finally:
            self.resources.close()

    def acquire_planner(self, env, seed, world, *cache_worlds, **options):
        if self.planner is None:
            self.planner = _create_motion_planner(env, seed, world, *cache_worlds, **options)
            self.planner_seed = seed
            self.planner.warmup(enable_graph=True, num_warmup_iterations=1)
        else:
            from curobo.types import ToolPoseCriteria
            from benchmark.backends.gsmini.workspace import update_collision_world

            self.planner.attachment_manager.detach()
            self.planner.update_tool_pose_criteria({"panda_hand": ToolPoseCriteria()})
            update_collision_world(self.planner, world)
            self.planner.reset_seed()
        return self.planner


def _peg_gripper_command(command, force, target_force, vertical_speed):
    """Reduce pinch pressure to allow gravity rotation; catch excessive slip."""
    import math

    if not all(math.isfinite(value) for value in (command, force, target_force, vertical_speed)):
        raise RuntimeError("Peg force control received non-finite telemetry")
    delta = -0.02 if vertical_speed < -0.005 else max(-0.005, min(0.005, (force - target_force) * 0.01))
    return max(-1.0, min(1.0, command + delta))


def _reorient_peg(env, app, action):
    """Hold the hand fixed while the peg swings under gravity, then lock it."""
    import math
    import torch
    from curobo.types import Pose
    from benchmark.backends.gsmini.execution import step_env

    obj = env.scene["object"].data
    point = env.peg_grasp_point_local
    tcp = env.scene["ee_frame"].data.target_pos_w[:, 0].clone()
    finger_ids, _ = env.scene["robot"].find_joints(env.cfg.actions.gripper_action.joint_names)
    command = env.scene["robot"].data.joint_pos[:, finger_ids].mean().item() / 0.02 - 1.0
    env.grasp_reference = None
    quiet = 0
    previous = None
    previous_pivot = None

    def pinch_forces():
        return [sum(env.scene[name].data.force_matrix_w.norm(dim=-1).sum().item()
                    for name in (f"gsmini_{side}_contact", f"gsmini_{side}_case_contact"))
                for side in ("left", "right")]

    for index in range(math.ceil(12.0 / env.step_dt)):
        pose = Pose(obj.root_pos_w, obj.root_quat_w)
        pivot = pose.transform_points(point)
        gap = torch.linalg.vector_norm(pivot - tcp).max().item()
        if not math.isfinite(gap) or gap > 0.025:
            raise RuntimeError(f"Peg escaped the controlled pinch: pivot drift={gap:.4f}m")
        forces = pinch_forces()
        # Low normal force permits a gravity-driven swing without changing friction.
        # Root descent is part of the intended swing. Detect slip at the pinch
        # point so gravity rotation is not mistaken for a falling grasp.
        pivot_speed = (0.0 if previous_pivot is None else
                       (pivot[0, 2] - previous_pivot[0, 2]).item() / env.step_dt)
        previous_pivot = pivot.clone()
        command = _peg_gripper_command(command, sum(forces) / 2, 0.04, pivot_speed)
        action[:, -1] = command
        step_env(env, app, action)
        axis = _rotation_pose(obj.root_quat_w).transform_points(
            torch.tensor([[0.0, 0.0, 1.0]], device=env.device))[0]
        aligned = axis[2].item() > 0.99
        stationary = previous is not None and torch.linalg.vector_norm(axis - previous).item() < 0.01
        quiet = quiet + 1 if aligned and stationary else 0
        previous = axis.clone()
        if index % 20 == 0:
            print(f"[REORIENT] axis_z={axis[2].item():.3f}, force={forces}, command={command:.4f}", flush=True)
        if quiet >= math.ceil(0.1 / env.step_dt):
            # Build a holding pinch without squeezing the soft pads fully shut.
            lock_steps = math.ceil(0.5 / env.step_dt)
            for _ in range(lock_steps):
                command = _peg_gripper_command(command, sum(pinch_forces()) / 2, 2.0, 0.0)
                action[:, -1] = command
                step_env(env, app, action)
            locked_axis = _rotation_pose(obj.root_quat_w).transform_points(
                torch.tensor([[0.0, 0.0, 1.0]], device=env.device))[0]
            if locked_axis[2].item() < 0.99:
                raise RuntimeError("Peg lost vertical alignment while locking the pinch")
            env.grasp_reference = _tcp_object_pose(env)
            return
    raise RuntimeError("Peg did not become vertical under gravity within 12s")


def _grasp_horizontal_peg(env, app, planner, action, center, quaternion):
    """Approach above the box, then servo short contact motions with physical drives."""
    import math
    import torch
    from benchmark.backends.gsmini.motion import _servo_hand_pose
    from benchmark.backends.gsmini.execution import hold
    from benchmark.backends.gsmini.workspace import grasp_poses

    above = center.clone()
    above[:, 2] += 0.16
    _move(env, app, planner, action, above, "box_approach", quaternion=quaternion)
    controller = _make_pose_controller(env)

    def track(destination):
        start = env.scene["ee_frame"].data.target_pos_w[:, 0].clone()
        steps = max(1, math.ceil(torch.linalg.vector_norm(destination - start).max().item() / 0.005))
        for index in range(1, steps + 1):
            point = start + (destination - start) * index / steps
            _servo_hand_pose(env, app, controller, action, grasp_poses(env, point, quaternion=quaternion))
        hold(env, app, action, 0.3)
        error = torch.linalg.vector_norm(env.scene["ee_frame"].data.target_pos_w[:, 0] - destination).max().item()
        if error > 0.005:
            raise RuntimeError(f"Box grasp servo blocked by physical contact: TCP error={error:.4f}m")

    track(center)
    action[:, -1] = -1.0
    hold(env, app, action, 0.5)
    env.grasp_reference = _tcp_object_pose(env)
    lift = env.scene["ee_frame"].data.target_pos_w[:, 0].clone()
    lift[:, 2] += 0.16
    track(lift)


def _sorting_slots(points, seed):
    from benchmark.tasks.recog.gsmini import _search_order

    slots = sorted(points, key=lambda name: tuple(round(v, 3) for v in points[name]))
    return slots, _search_order(len(slots), seed)


def _search_sorting_ball(resources, env, app, planner, action, task, seed):
    """Probe randomized physical slots, retaining rejected grasps in the episode."""
    from curobo.types import Pose
    from benchmark.backends.gsmini.motion import _servo_hand_pose
    from benchmark.catalog import candidate_names as _candidate_names
    from benchmark.backends.gsmini.geometry import _object_center_local
    from benchmark.tasks.recog.gsmini import _candidate_binding, _configure_search_recording, _grasp_with_contact, _release_search_candidate, _search_until_match, _verify_candidate_contact
    from benchmark.backends.gsmini.execution import hold
    from benchmark.backends.gsmini.workspace import grasp_poses, read_collision_world, update_collision_world

    names = _candidate_names(task)
    target = task["obj_of_interest"][0]
    points = {}
    for name in task["tactile_targets"]:
        with _candidate_binding(env, name, target):
            points[name] = _geometric_grasp_position(env, dict(task, obj_of_interest=[name]))[0, :2].tolist()
    slots, order = _sorting_slots(points, seed)
    attempts = ExitStack()
    resources.callback(attempts.close)
    candidate_forces = _configure_search_recording(env, names, slots)

    def probe(slot):
        name = slots[slot]
        attempts.enter_context(_candidate_binding(env, name, target))
        candidate_task = dict(task, obj_of_interest=[name])
        env.search_slot, env.search_phase, env.search_match = slot, 0, -1
        action[:, -1] = 1.0
        update_collision_world(planner, read_collision_world(env, ("Table", *(n for n in task["objects"] if n != name))))
        center = _geometric_grasp_position(env, candidate_task)
        overhead = center.clone()
        overhead[:, 2] += 0.10
        _move(env, app, planner, action, overhead, "search_approach")
        env.search_phase = 1
        _grasp_with_contact(env, app, action, candidate_task, grasp_height=0.008)
        _verify_candidate_contact(candidate_forces(slot))
        # Same post-contact metadata teacher as the blind expert, not a tactile classifier.
        matched = name == target
        env.search_match, env.search_phase = int(matched), 3 if matched else 2
        hold(env, app, action, env.step_dt)
        print(f"[SEARCH] slot={slot}, bilateral contact, match={matched}", flush=True)
        return matched

    release = partial(_release_search_candidate, env, app, planner, action, attempts)

    _search_until_match(order, probe, release)
    env.grasp_center_local = _object_center_local(env, task)[None]
    env.grasp_reference = _tcp_object_pose(env)
    source_height = Pose(env.scene["object"].data.root_pos_w,
                         env.scene["object"].data.root_quat_w).transform_points(
                             env.grasp_center_local)[:, 2].clone()
    tcp = env.scene["ee_frame"].data
    start = tcp.target_pos_w[:, 0].clone()
    lift = start.clone()
    lift[:, 2] += 0.14
    controller = _make_pose_controller(env)
    quaternion = tcp.target_quat_w[0, 0].tolist()
    for index in range(1, 71):
        point = start + (lift - start) * index / 70
        _servo_hand_pose(env, app, controller, action, grasp_poses(env, point, quaternion=quaternion))
    return source_height


def _planner_velocity_scale(task_id: int) -> float:
    if task_id in (49, 51):
        return 0.5
    return 1.0


def _configure_bulb_gripper(cfg, task):
    object_name = task["obj_of_interest"][0]
    if task["objects"][object_name]["type"] != "tactile_bulb":
        return
    command = {"panda_finger_joint.*": float(task["reset_contract"]["gripper_joint_pos"])}
    cfg.actions.gripper_action.close_command_expr = command.copy()


def _retreat_collision_objects(obstacles, target_name: str, object_name: str):
    """Include all fixtures and the released object in the retract plan."""
    return ("Table", *dict.fromkeys((*obstacles, target_name, object_name)))


def _align_above_bore(track, measure, max_corrections: int = 8,
                      radial_clearance=BORE_RADIAL_CLEARANCE):
    """Close small bore-entry residuals with measured IK corrections."""
    for correction in range(max_corrections + 1):
        radial, axial, angle = measure()
        if radial <= radial_clearance and axial <= 0.0005 and angle <= 0.002:
            return radial, axial, angle
        if correction == max_corrections:
            raise RuntimeError(
                "Bulb cannot align within the measured bore clearance: "
                f"radial={radial:.5f}m, axial={axial:.5f}m, angle={angle:.3f}rad"
            )
        print(f"[THREAD] precision bore correction {correction + 1}: "
              f"radial={radial:.5f}m, axial={axial:.5f}m, angle={angle:.3f}rad",
              flush=True)
        track(correction + 1)


@contextmanager
def _precise_bore_ik(planner):
    """Tighten cuRobo's early-success gate only for narrow bore tracking."""
    config = planner.ik_solver.config
    previous = config.position_tolerance, config.orientation_tolerance
    config.position_tolerance = 0.0001
    config.orientation_tolerance = 0.001
    try:
        yield
    finally:
        config.position_tolerance, config.orientation_tolerance = previous


def _bore_staging_pose(env, goal, object_goal, target_name):
    """Measure the existing inclined-bore staging goal in world coordinates."""
    import torch
    from curobo.types import Pose

    target = env.scene[target_name].data
    entry_local = torch.tensor([goal["insertion_entry_local"]],
                               device=env.device, dtype=object_goal.position.dtype)
    entry = Pose(target.root_pos_w, target.root_quat_w).transform_points(entry_local)
    axis_local = torch.zeros_like(object_goal.quaternion[..., :3])
    axis_local[..., 2] = 1.0
    axis = _rotation_pose(object_goal.quaternion).transform_points(axis_local)
    return Pose(entry + 0.05 * axis, object_goal.quaternion), axis


def _geometric_grasp_position(env, task: dict):
    """Aim at an authored grasp point or the asset's physical bounds center."""
    import torch
    from curobo.types import Pose

    name = task["obj_of_interest"][0]
    spec = task["objects"][name]
    grasp_point = (task.get("reset_contract") or {}).get("grasp_point_local")
    if grasp_point is None:
        minimum, maximum, _ = _asset_geometry(spec["type"])
        grasp_point = [(minimum[axis] + maximum[axis]) * 0.5 for axis in range(3)]
        if (spec["type"] == "test_tube_insert"
                and task.get("episode_config", {}).get("expert_family") == "tube"):
            # Leave the 130 mm tube's top clear of the GSmini palm (52 mm above TCP).
            grasp_point[2] = 0.085
    center = torch.tensor(grasp_point, device=env.device,
                          dtype=env.scene["object"].data.root_pos_w.dtype)
    center *= torch.tensor(spec["scale"], device=env.device, dtype=center.dtype)
    if tabletop_variant(task) == 48:
        # Leave only 7 mm above the pinch so the swinging tip clears the sensor case.
        center[2] = 0.099 * spec["scale"][2]
        center[1] += 0.006
        env.peg_grasp_point_local = center[None].clone()
    elif tabletop_variant(task) == 49:
        # Pinch the upper body with the sensor case clear of the box floor.
        center[2] += 0.010
    elif (spec["type"] == "gear_large"
          or (spec["type"] == "gear_medium"
              and TABLETOP_IDENTITIES.get(tabletop_variant(task), (None, None))[0] == "tac_extra")):
        # Pinch the upper rim symmetrically; an off-center circular pinch ejects the gear.
        center[2] += 0.025
    elif spec["type"] == "key_insert":
        # Pinch the exposed head; the blade center lies inside the source lock.
        center[:] = 0.0
    elif spec["type"] == "hdmi_connector":
        # Pinch the upper body with clearance from the tabletop around its base.
        center[2] += 0.008
    obj = env.scene["object"].data
    return Pose(obj.root_pos_w, obj.root_quat_w).transform_points(center.expand_as(obj.root_pos_w))


def _verify_uncontacted_object(env, initial_position, initial_quaternion, phase: str):
    """Reject a pre-grasp route that physically disturbed its target."""
    import math
    import torch
    from curobo.types import Pose

    obj = env.scene["object"].data
    displacement = torch.linalg.vector_norm(obj.root_pos_w - initial_position, dim=-1).max().item()
    rotation = Pose(quaternion=obj.root_quat_w).angular_distance(
        Pose(quaternion=initial_quaternion)).max().item()
    if (not math.isfinite(displacement) or not math.isfinite(rotation)
            or displacement > 0.005 or rotation > 0.1):
        raise RuntimeError(f"{phase} moved bulb before grasp: "
                           f"position={displacement:.4f}m, rotation={rotation:.3f}rad")


def _threaded_object_quaternion(initial_quaternion, yaw_rad: float):
    """Rotate the grasped object around its local insertion axis."""
    from curobo.types import Pose

    euler = initial_quaternion.new_zeros((*initial_quaternion.shape[:-1], 3))
    euler[..., 2] = yaw_rad
    return _rotation_pose(initial_quaternion).multiply(Pose.from_euler_xyz(euler)).quaternion


def _threaded_path_pose(goal, yaw, total_yaw, pitch):
    """Follow the source's right-handed helix in the socket frame."""
    import math
    import torch
    from curobo.types import Pose

    offset = torch.zeros_like(goal.position)
    offset[:, 2] = (yaw - total_yaw) * pitch / (2 * math.pi)
    return Pose(goal.transform_points(offset),
                _threaded_object_quaternion(goal.quaternion, yaw))


def _thread_progress_height(planned, measured, advance):
    """Preserve the 2 mm pitch without lifting the bulb after it settles deeper."""
    import math

    return max(0.0, min(planned - max(0.0, advance) * 0.002 / (2 * math.pi), measured))


def _threaded_seated(env, task, initial_quaternion):
    """Stop turning inside the original goal; release and stability checks follow."""
    return check_task_success(
        env, task, suite=TABLETOP_IDENTITIES[tabletop_variant(task)][0],
        initial_quaternion=initial_quaternion)["success"]


def _unwrap_thread_angle(measured, reference):
    import math

    return reference + math.atan2(math.sin(measured - reference), math.cos(measured - reference))


def _thread_rotation_command(observed, end, lead, bias):
    """Correct bounded angular lag using the measured screw phase."""
    if observed <= end + lead:
        bias = max(-0.25, min(0.0, bias + 0.2 * (end - observed)))
    return max(end, observed - lead) + bias, bias


def _thread_strokes(total_yaw, max_stroke):
    """Partition a clockwise helix into contiguous bounded wrist rotations."""
    import math

    if not math.isfinite(total_yaw) or total_yaw >= 0 or not math.isfinite(max_stroke) or max_stroke <= 0:
        raise ValueError("Thread strokes require negative total yaw and positive stroke length")
    count = math.ceil(abs(total_yaw) / max_stroke)
    return [(max(total_yaw, -index * max_stroke), max(total_yaw, -(index + 1) * max_stroke))
            for index in range(count)]


def _thread_grasp(env, *, follow_translation=True, axial_correction_limit=0.0):
    """Track the pinch point while preserving the captured torque reference."""
    grasp = env.grasp_reference.clone()
    if follow_translation:
        grasp.position = -_rotation_pose(grasp.quaternion).transform_points(
            _tcp_object_pose(env).inverse().position)
    elif axial_correction_limit:
        # Correct only bounded axial pad slip; retain lateral and torque references.
        point = grasp.inverse().position
        measured = _tcp_object_pose(env).inverse().position
        point[:, 2] += (measured[:, 2] - point[:, 2]).clamp(
            -axial_correction_limit, axial_correction_limit)
        grasp.position = -_rotation_pose(grasp.quaternion).transform_points(point)
    return grasp


def _execute_threaded_insertion(env, app, planner, action, engage_pose,
                                target_position, initial_quaternion,
                                angular_speed=0.5, goal_for_duration=None,
                                tracking_horizon=None, regrasp=None, success_check=None,
                                engaged_angular_speed=None,
                                engaged_max_joint_step=0.2, initial_max_joint_step=0.2, contact_progress=False,
                                follow_grasp_translation=True, rotation_lead_limit=0.3,
                                grasp_axial_correction=0.0):
    """Center above the bore, then follow the screw path with bounded local IK."""
    import math
    import torch
    from curobo.types import GoalToolPose, JointState, Pose
    from benchmark.backends.gsmini.execution import step_env

    yaw = -(4 * 2 * math.pi + 2.0)
    pitch = 0.002
    horizon = env.step_dt if tracking_horizon is None else tracking_horizon
    start = Pose(env.scene["object"].data.root_pos_w.clone(),
                 env.scene["object"].data.root_quat_w.clone())
    if regrasp is not None:
        env.grasp_reference = _tcp_object_pose(env)
        print(f"[THREAD] captured grasp={env.grasp_reference.inverse().position.tolist()}, "
              f"quaternion={env.grasp_reference.inverse().quaternion.tolist()}", flush=True)
    base_goal = Pose(target_position, initial_quaternion)
    engage_relative = base_goal.inverse().multiply(engage_pose)
    approach_relative = base_goal.inverse().multiply(start)

    def current_engage(duration=0.0):
        return (engage_pose if goal_for_duration is None else
                goal_for_duration(duration).multiply(engage_relative))
    robot = env.scene["robot"]
    arm = env.action_manager.get_term("arm_action")
    arm_ids, arm_names = robot.find_joints(arm.cfg.joint_names,
                                           preserve_order=arm.cfg.preserve_order)
    contact_links = planner.kinematics.config.kinematics_config.grasp_contact_link_names or ()
    planner.disable_link_collision(contact_links)
    tracking_bias = torch.zeros_like(start.position)
    tilt_bias = torch.zeros_like(start.quaternion)
    tilt_bias[:, 0] = 1.0
    stroke = 0
    try:
        if regrasp is not None:
            from benchmark.backends.gsmini.motion import _servo_hand_pose
            controller = _make_pose_controller(env)

        def track_pose(desired, phase, index):
            if regrasp is not None:
                local_axis = desired.position.new_tensor([[0.0, 0.0, 1.0]])
                wanted_axis = _rotation_pose(desired.quaternion).transform_points(local_axis)
                actual_axis = _rotation_pose(env.scene["object"].data.root_quat_w).transform_points(local_axis)
                correction = 0.15 * torch.cross(actual_axis, wanted_axis, dim=-1)
                correction = torch.cat((torch.ones_like(correction[:, :1]), correction), dim=-1)
                correction /= correction.norm(dim=-1, keepdim=True)
                tilt_bias.copy_(_rotation_pose(correction).multiply(_rotation_pose(tilt_bias)).quaternion)
                vector = tilt_bias[:, 1:]
                vector *= math.sin(0.125) / vector.norm(dim=-1, keepdim=True).clamp_min(math.sin(0.125))
                tilt_bias[:, :1] = torch.sqrt(1.0 - vector.square().sum(dim=-1, keepdim=True))
                desired = Pose(desired.position + tracking_bias,
                               _rotation_pose(tilt_bias).multiply(_rotation_pose(desired.quaternion)).quaternion)
                if phase == "screw insertion":
                    goal = goal_for_duration(horizon)
                    axis = _rotation_pose(goal.quaternion).transform_points(
                        desired.position.new_tensor([[0.0, 0.0, 1.0]]))
                    height = ((desired.position - goal.position) * axis).sum(dim=-1, keepdim=True)
                    desired.position -= height.clamp_max(0.0) * axis
            grasp = None
            if regrasp is not None and phase not in ("bore alignment", "precision bore alignment"):
                grasp = _thread_grasp(env, follow_translation=follow_grasp_translation,
                                      axial_correction_limit=grasp_axial_correction)
            hand_goal = _object_pose_hand_goal(env, desired, grasp=grasp)
            if regrasp is not None:
                if contact_progress and phase == "screw insertion":
                    current = _planner_joint_state(robot, planner)
                    result = planner.ik_solver.solve_pose(
                        GoalToolPose.from_poses({"panda_hand": hand_goal}), current,
                        return_seeds=planner.ik_solver.config.num_seeds)
                    candidates = result.solution.reshape(env.num_envs, -1, len(planner.joint_names))
                    try:
                        solution = _nearest_joint_solution(
                            candidates, result.success.reshape(env.num_envs, -1), current.position,
                            limits=robot.data.soft_joint_pos_limits[:, arm_ids], joint_margin=0.05)
                    except RuntimeError:
                        pass
                    else:
                        joints = JointState.from_position(
                            solution, joint_names=planner.joint_names).reorder(arm_names).position
                        action[:, :arm.action_dim] = arm.position_actions(joints)
                        step_env(env, app, action)
                        return
                _servo_hand_pose(env, app, controller, action, hand_goal,
                                 center_joints=True, max_joint_step=(
                                     0.2 if phase != "screw insertion" else
                                     initial_max_joint_step if stroke == 0 else engaged_max_joint_step),
                                 joint_margin=0.05 if contact_progress else 0.0)
                return
            goal = GoalToolPose.from_poses({"panda_hand": hand_goal})
            current = _planner_joint_state(robot, planner)
            result = planner.ik_solver.solve_pose(
                goal, current,
                return_seeds=planner.ik_solver.config.num_seeds,
            )
            success = result.success.reshape(env.num_envs, -1)
            if not success.any(dim=-1).all():
                if phase not in ("bore alignment", "precision bore alignment"):
                    raise RuntimeError(f"Thread control lost local IK during {phase} step {index}")
                print(f"[THREAD] IK had no valid local solution during {phase} step {index}; "
                      "replanning this pose with cuRobo MotionPlanner", flush=True)
                planner.enable_link_collision(contact_links)
                _move_object_pose(env, app, planner, action, desired, "insertion")
                planner.disable_link_collision(contact_links)
                return
            candidates = result.solution.reshape(
                env.num_envs, -1, len(planner.joint_names),
            )
            solution = _nearest_joint_solution(candidates, success, current.position)
            joints = JointState.from_position(
                solution, joint_names=planner.joint_names,
            ).reorder(arm_names).position
            action[:, :arm.action_dim] = arm.position_actions(joints[:, :len(arm_ids)])
            step_env(env, app, action)

        align_error = torch.linalg.vector_norm(engage_pose.position - start.position).max().item()
        align_angle = start.angular_distance(engage_pose).max().item()
        align_steps = max(1, math.ceil(
            max(align_error / (0.02 if regrasp is not None else 0.005), align_angle / 0.1) / env.step_dt,
        ))
        print(f"[THREAD] centering above bore: {align_steps} steps", flush=True)
        for index in range(1, align_steps + 1):
            fraction = index / align_steps
            desired = Pose(
                start.position + fraction * (engage_pose.position - start.position),
                engage_pose.quaternion,
            )
            if goal_for_duration is not None:
                future = goal_for_duration(horizon)
                desired = Pose(future.transform_points(
                    approach_relative.position + fraction * (
                        engage_relative.position - approach_relative.position)),
                    current_engage(horizon).quaternion)
            track_pose(desired, "bore alignment", index)

        def measure_alignment():
            aligned = Pose(env.scene["object"].data.root_pos_w,
                           env.scene["object"].data.root_quat_w)
            engage = current_engage()
            radial, axial = _insertion_errors(
                env.task["goals"][0], engage,
                aligned.position[0] - engage.position[0],
            )
            angle = aligned.angular_distance(engage).max().item()
            if regrasp is not None:
                relative = engage.inverse().multiply(aligned).quaternion[0]
                angle = 2 * math.atan2(relative[1:3].norm().item(), relative[[0, 3]].norm().item())
            return radial.item(), axial.item(), angle

        with _precise_bore_ik(planner):
            def correct_bore(index):
                if regrasp is not None:
                    tracking_bias.add_(0.2 * (current_engage().position
                                             - env.scene["object"].data.root_pos_w))
                    tracking_bias.clamp_(-0.003, 0.003)
                track_pose(current_engage(horizon), "precision bore alignment", index)

            _align_above_bore(correct_bore, measure_alignment,
                             max_corrections=40 if regrasp is not None else 8,
                             radial_clearance=0.00002 if regrasp is not None else BORE_RADIAL_CLEARANCE)
            aligned = Pose(
                env.scene["object"].data.root_pos_w.clone(),
                env.scene["object"].data.root_quat_w.clone(),
            )
            if regrasp is not None:
                env.grasp_reference = _tcp_object_pose(env)
                tracking_bias.zero_()
                tilt_bias.zero_()
                tilt_bias[:, 0] = 1.0

            thread_height = -yaw * pitch / (2 * math.pi)
            progress_yaw = 0.0

            def screw_goal(angle, duration=0.0):
                goal = base_goal if goal_for_duration is None else goal_for_duration(duration)
                if contact_progress:
                    offset = torch.zeros_like(goal.position)
                    offset[:, 2] = max(0.0, thread_height + (angle - progress_yaw) * pitch / (2 * math.pi))
                    return Pose(goal.transform_points(offset),
                                _threaded_object_quaternion(goal.quaternion, angle))
                return _threaded_path_pose(goal, angle, yaw, pitch)

            def measured_rotation(reference):
                nonlocal thread_height, progress_yaw
                actual = Pose(env.scene["object"].data.root_pos_w,
                              env.scene["object"].data.root_quat_w)
                goal = base_goal if goal_for_duration is None else goal_for_duration(0.0)
                local = goal.inverse().multiply(actual)
                relative = local.quaternion[0]
                measured = 2 * math.atan2(relative[3].item(), relative[0].item())
                measured = _unwrap_thread_angle(measured, reference)
                if contact_progress:
                    thread_height = _thread_progress_height(
                        thread_height, local.position[0, 2].item(), progress_yaw - measured)
                    progress_yaw = measured
                return relative, measured

            entry = ((base_goal if goal_for_duration is None else goal_for_duration(horizon))
                     if contact_progress else screw_goal(0.0, horizon))
            entry_steps = max(1, math.ceil(
                torch.linalg.vector_norm(aligned.position - entry.position).max().item()
                / ((0.02 if contact_progress else 0.005) * env.step_dt)))
            entry_frame = base_goal if goal_for_duration is None else goal_for_duration(0.0)
            entry_relative = entry_frame.inverse().multiply(aligned)
            entry_offset = (torch.zeros_like(entry_relative.position) if contact_progress else
                            entry_frame.inverse().multiply(screw_goal(0.0)).position)
            previous_height = entry_relative.position[0, 2].item()
            stalled = 0
            for index in range(1, entry_steps + 1):
                fraction = index / entry_steps
                goal = base_goal if goal_for_duration is None else goal_for_duration(horizon)
                desired = Pose(goal.transform_points(
                    entry_relative.position * (1 - fraction) + entry_offset * fraction),
                    goal.quaternion)
                track_pose(desired, "thread lead-in", index)
                if contact_progress:
                    actual = goal.inverse().multiply(Pose(
                        env.scene["object"].data.root_pos_w, env.scene["object"].data.root_quat_w))
                    commanded = entry_relative.position[0, 2].item() * (1 - fraction)
                    height = actual.position[0, 2].item()
                    stalled = stalled + 1 if height - commanded > 0.001 and abs(height - previous_height) < 0.0002 else 0
                    previous_height = height
                    if stalled >= 2:
                        print(f"[THREAD] measured contact entry at {height:.5f}m", flush=True)
                        break
            if contact_progress:
                thread_height = max(0.0, previous_height)
                env.grasp_reference = _tcp_object_pose(env)

            target_pose = screw_goal(yaw)
            strokes = [(0.0, yaw)] if regrasp is None or contact_progress else _thread_strokes(yaw, 3.5)
            for stroke, (start_yaw, end_yaw) in enumerate(strokes):
                stroke_speed = (angular_speed if stroke == 0 or engaged_angular_speed is None
                                else engaged_angular_speed)
                steps = max(1, math.ceil(abs(end_yaw - start_yaw) / (stroke_speed * env.step_dt)))
                print(f"[THREAD] stroke {stroke + 1}/{len(strokes)}: "
                      f"{start_yaw:.3f} -> {end_yaw:.3f}rad, {steps} steps", flush=True)
                if regrasp is None:
                    for index in range(1, steps + 1):
                        angle = start_yaw + (end_yaw - start_yaw) * index / steps
                        track_pose(screw_goal(angle, horizon), "screw insertion", index)
                else:
                    observed = start_yaw
                    angular_bias = 0.0
                    for index in range(1, 8 * steps + 1):
                        relative, observed = measured_rotation(observed)
                        angle, angular_bias = _thread_rotation_command(
                            observed, end_yaw, min(stroke_speed * horizon, rotation_lead_limit), angular_bias)
                        desired = screw_goal(angle, horizon)
                        track_pose(desired, "screw insertion", index)
                        if success_check is not None and success_check():
                            return
                        if contact_progress:
                            joints = robot.data.joint_pos[:, arm_ids]
                            limits = robot.data.soft_joint_pos_limits[:, arm_ids]
                            margin = torch.minimum(joints - limits[:, :, 0], limits[:, :, 1] - joints).min().item()
                            if margin < 0.15:
                                if start_yaw - observed < 0.5 or len(strokes) >= 10:
                                    raise RuntimeError("Thread rotation exhausted its safe joint travel")
                                end_yaw = observed
                                angular_bias = 0.0
                                strokes.append((end_yaw, yaw))
                                _hold_current_arm_action(env, action)
                                print(f"[THREAD] rewinding at joint margin={margin:.3f}rad", flush=True)
                                break
                        if index % 20 == 0:
                            tilt = 2 * math.atan2(relative[1:3].norm().item(), relative[[0, 3]].norm().item())
                            print(f"[THREAD] measured yaw={observed:.3f}rad, tilt={tilt:.3f}rad, "
                                  f"wrist={robot.data.joint_pos[0, arm_ids[-1]].item():.3f}rad", flush=True)
                        if observed <= end_yaw + 0.03:
                            break
                    else:
                        raise RuntimeError(f"Thread rotation stalled at {observed:.3f}rad before {end_yaw:.3f}rad")
                if regrasp is not None:
                    for index in range(40):
                        _, observed = measured_rotation(end_yaw)
                        angle, angular_bias = _thread_rotation_command(
                            observed, end_yaw, min(stroke_speed * horizon, rotation_lead_limit), angular_bias)
                        desired = screw_goal(angle, horizon)
                        track_pose(desired, "screw insertion", index)
                        if success_check is not None and success_check():
                            return
                        actual = Pose(env.scene["object"].data.root_pos_w,
                                      env.scene["object"].data.root_quat_w)
                        if actual.angular_distance(screw_goal(end_yaw)).max().item() < 0.03:
                            break
                if stroke + 1 < len(strokes):
                    actual = Pose(env.scene["object"].data.root_pos_w,
                                  env.scene["object"].data.root_quat_w)
                    target = screw_goal(end_yaw)
                    print(f"[THREAD] stroke endpoint error={actual.angular_distance(target).max().item():.3f}rad, "
                          f"position_error={(actual.position - target.position).tolist()}", flush=True)
                    regrasp(start_yaw - end_yaw)
                    tilt_bias.zero_()
                    tilt_bias[:, 0] = 1.0
    finally:
        planner.enable_link_collision(contact_links)

    actual = Pose(env.scene["object"].data.root_pos_w,
                  env.scene["object"].data.root_quat_w)
    if goal_for_duration is not None:
        future = goal_for_duration(0.0)
        target_pose = Pose(future.position, _threaded_object_quaternion(future.quaternion, yaw))
    radial, axial = _insertion_errors(
        env.task["goals"][0], target_pose,
        actual.position[0] - target_pose.position[0],
    )
    angle = actual.angular_distance(target_pose).max().item()
    if radial.item() > 0.003 or axial.item() > 0.003 or angle > 0.05:
        raise RuntimeError(
            f"cuRobo IK insertion endpoint error: radial={radial.item():.4f}m, "
            f"axial={axial.item():.4f}m, angle={angle:.3f}rad"
        )


def _bulb_grasp_quaternion(object_quaternion):
    """Keep the downward grasp while aligning the finger gap to the bulb cross-section."""
    import math
    import torch
    from curobo.types import Pose

    position = torch.zeros((*object_quaternion.shape[:-1], 3),
                           device=object_quaternion.device, dtype=object_quaternion.dtype)
    radial_axis = torch.zeros_like(position)
    radial_axis[..., 1] = 1.0
    radial_axis = Pose(position, object_quaternion).transform_points(radial_axis)
    yaw = torch.atan2(radial_axis[..., 0], -radial_axis[..., 1])
    euler = torch.zeros_like(position)
    euler[..., 0] = math.pi
    euler[..., 2] = yaw
    return Pose.from_euler_xyz(euler).quaternion


def _in_hand_object_pose(tcp_position, tcp_quaternion, local_quaternion,
                         grasp_point_local, scale):
    """Place the asset's authored grasp point at the GelSight pad midpoint."""
    import torch
    from curobo.types import Pose

    object_quaternion = _rotation_pose(tcp_quaternion).multiply(
        _rotation_pose(local_quaternion)).quaternion
    grasp_offset = Pose(torch.zeros_like(tcp_position), object_quaternion).transform_points(
        grasp_point_local * scale)
    return tcp_position - grasp_offset, object_quaternion


def resolve_tabletop_task(task):
    if tabletop_variant(task) != 51:
        return task
    contract = dict(task["reset_contract"], fingertip_height=0.22)
    if "grasp_point_local" not in contract:
        if "grasp_center_local_xy" in contract:
            xy = contract["grasp_center_local_xy"]
        else:
            spec = task["objects"][task["obj_of_interest"][0]]
            _, _, center = _asset_geometry(spec["type"])
            xy = center[:2]
        contract["grasp_point_local"] = [*xy, 0.10]
    return dict(task, reset_contract=contract)


def tabletop_joint_defaults(task):
    if tabletop_variant(task) == 49:
        # Prepare the arm above both randomized search slots before the falling
        # battery settles; the low source pose can intersect its upper body.
        task = dict(task, reset_contract=dict(task["reset_contract"], fingertip_height=0.22))
    return robot_joint_defaults(resolve_tabletop_task(task))


def _initialize_in_hand(env, task: dict):
    """Apply the documented grasp as an episode reset, before recording actions."""
    import torch
    from isaaclab.utils.math import quat_from_euler_xyz

    contract = task["reset_contract"]
    obj = env.scene["object"]
    tcp = env.scene["ee_frame"].data
    object_name = task["obj_of_interest"][0]
    spec = task["objects"][object_name]
    tcp_quat = tcp.target_quat_w[:, 0]
    dtype = obj.data.root_pos_w.dtype
    device = env.device
    if "grasp_point_local" in contract:
        grasp_point = contract["grasp_point_local"]
    else:
        _, _, grasp_point = _asset_geometry(spec["type"])
    if "grasp_center_local_xy" in contract:
        grasp_point = (*contract["grasp_center_local_xy"], grasp_point[2])
    grasp_local = torch.tensor(grasp_point, device=device, dtype=dtype).expand(env.num_envs, 3)
    scale = torch.tensor(spec["scale"], device=device, dtype=dtype).expand(env.num_envs, 3)
    local_quaternion = torch.tensor(
        contract.get("in_hand_object_local_quat_wxyz", (0.0, 0.0, 1.0, 0.0)),
        device=device, dtype=tcp_quat.dtype,
    ).expand_as(tcp_quat)
    yaw = torch.full((env.num_envs,), float(contract.get("grasp_yaw_rad", 0.0)),
                     device=device, dtype=tcp_quat.dtype)
    local_yaw = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    local_quaternion = _rotation_pose(local_quaternion).multiply(
        _rotation_pose(local_yaw)).quaternion
    center, object_quat = _in_hand_object_pose(
        tcp.target_pos_w[:, 0],
        tcp_quat, local_quaternion, grasp_local, scale,
    )
    obj.write_root_pose_to_sim(torch.cat((center, object_quat), dim=-1))
    obj.write_root_velocity_to_sim(torch.zeros((env.num_envs, 6), device=env.device))
    return object_quat.clone()


def _check_insertion_motion(env, task: dict, target_name: str):
    """Abort a descending trajectory on lateral drift or loss of the physical grasp."""
    import torch

    goal = task["goals"][0]
    xy_threshold = goal.get("thread_engage_xy_threshold", goal["xy_threshold"])
    position = env.scene["object"].data.root_pos_w[0]
    desired_pose = _task_object_goal(env, task, target_name)
    desired = desired_pose.position[0]
    delta = position - desired
    lateral_error, _ = _insertion_errors(goal, desired_pose, delta)
    opening = _gripper_opening(env)[0]
    motion_xy_threshold = getattr(env, "insertion_motion_xy_threshold", None)
    if motion_xy_threshold is None:
        motion_xy_threshold = xy_threshold
    if "thread_engagement_range" in goal:
        rim_world = getattr(env, "thread_rim_world", None)
        axis_world = getattr(env, "thread_axis_world", None)
        if rim_world is not None and axis_world is not None:
            rim_clearance = torch.dot(
                position - rim_world[0], axis_world[0]).item()
        else:
            rim_clearance = 0.0
        if rim_clearance > 0.003:
            motion_xy_threshold = max(motion_xy_threshold, 0.02)
        else:
            motion_xy_threshold = min(
                motion_xy_threshold, goal.get("thread_engage_xy_threshold", xy_threshold))
    valid = (torch.isfinite(torch.cat((position, desired, opening.reshape(1)))).all()
             & (lateral_error <= motion_xy_threshold)
             & (opening > 0.001))
    if getattr(env, "grasp_reference", None) is None:
        tcp = env.scene["ee_frame"].data.target_pos_w[0, 0]
        valid &= torch.linalg.vector_norm(tcp - position) <= 0.045
    if not valid.item():
        radial = lateral_error.item()
        gap = torch.linalg.vector_norm(
            env.scene["ee_frame"].data.target_pos_w[0, 0] - position).item()
        raise RuntimeError(
            f"Insertion stopped: radial={radial:.4f}m/{motion_xy_threshold:.4f}m "
            f"(final tolerance={xy_threshold:.4f}m), "
            f"xyz={delta.tolist()}, "
            f"gripper_opening={opening.item():.4f}m, tcp_gap={gap:.4f}m, "
            f"grasp_reference={getattr(env, 'grasp_reference', None) is not None}"
        )


def settle_tabletop_scene(env, app, action, task: dict) -> None:
    """Reuse collection settling parameters and checks during policy evaluation."""
    from benchmark.backends.gsmini.execution import settle_object

    task_id = tabletop_variant(task)
    expert_family = task["episode_config"]["expert_family"]
    tube_task = expert_family == "tube"
    if task_id == 50:
        from benchmark.backends.gsmini.execution import settle_scene as _settle_scene
        _settle_scene(env, app, action, task)
    elif expert_family == "key":
        settle_object(env, app, action, max_displacement=0.06)
    elif task_id == 52:
        settle_object(env, app, action, max_seconds=5.0,
                      max_angular_speed=0.1, require_velocity_quiet=True)
        _verify_pose_stability(
            env, app, action, context="Bulb source before grasp",
            max_position_drift=0.0005, max_orientation_drift=0.01,
            max_wait_seconds=10.0,
        )
    else:
        source_name = task["obj_of_interest"][0].replace("test_tube_insert_", "test_tube_base_source_")
        settle_object(
            env, app, action,
            max_seconds=5.0 if tube_task else 1.5,
            # The tube's reset deliberately suspends it above its source.
            # Bound its motion around the support, rather than counting
            # the authored drop as motion away from the tabletop.
            initial_position=(env.scene[source_name].data.root_pos_w
                              if tube_task else None),
            max_displacement=0.05 if tube_task else 0.03,
            require_drop=False,
        )


def make_tabletop_environment_config(seed: int, task: dict):
    """Build the same task configuration for expert collection and policy evaluation."""
    task = resolve_tabletop_task(task)
    cfg = _make_env_cfg(seed, task)
    if task.get("episode_config", {}).get("expert_family") == "tube":
        # Resolve the light tube's clamped contacts without alternating-step slip.
        cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 4
    cfg.scene.wristview.update_latest_camera_pose = False
    if tabletop_variant(task) == 52:
        cfg.episode_length_s = 180.0
    _configure_bulb_gripper(cfg, task)
    return cfg


def run(app, seed: int, output: Path, task_id: int = 0, *,
        session: TubeCollectionSession | GSminiCollectionSession | None = None):
    if type(task_id) is not int or task_id not in TABLETOP_IDENTITIES:
        raise ValueError(f"Expert unavailable for tabletop strategy {task_id}")
    suite, public_task_id = TABLETOP_IDENTITIES[task_id]
    short_suite = suite.removeprefix("tac_")
    gsmini_base0 = task_id == 0
    gsmini_threading = task_id == 52
    if session is not None and public_task_id not in REUSABLE_GSMINI_TASK_IDS.get(short_suite, ()):
        raise ValueError("Collection session is unavailable for this tabletop strategy")
    if isinstance(session, GSminiCollectionSession) and session.task_key != (short_suite, public_task_id):
        raise ValueError("Collection session cannot change tasks")
    if session is not None and not gsmini_base0 and not isinstance(session, GSminiCollectionSession):
        raise ValueError("Search and sorting require a task-specific collection session")
    _prepare_output_path(output)

    import math
    import torch
    from curobo.types import Pose
    from isaaclab.envs import ManagerBasedRLEnv
    from benchmark.backends.gsmini.execution import configure_shutdown, execute_trajectory, hold, plan_grasp, settle_object, step_env, verify_grasp
    from benchmark.backends.gsmini.workspace import read_collision_world, update_collision_world

    task = get_task(suite, public_task_id)
    task = resolve_tabletop_task(task)
    goal = task["goals"][0]
    expert_family = task.get("episode_config", {}).get("expert_family")
    adaptive_insertion = expert_family in ("hdmi", "gear") or task_id in (48, 51)
    adaptive_motion = expert_family in SHARED_TABLETOP_EXPERT_FAMILIES or task_id in (48, 49, 50, 51)
    angled_insertion = "insertion_entry_local" in goal
    object_name, target_name = task["obj_of_interest"][0], task["targets"][0]
    in_hand = (task.get("reset_contract") or {}).get("mode") == "in_hand"
    obstacles = tuple(name for name in task["objects"]
                      if name != object_name and (not in_hand or name != target_name))

    class RecordingEnv(ManagerBasedRLEnv):
        episode = None
        inserting = False
        task = None
        target_name = None
        grasp_reference = None
        thread_rim_world = None
        thread_axis_world = None
        expert_success_monitor = None

        def step(self, action):
            _prepare_recording_step(self, action)
            result = super().step(action)
            if self.grasp_reference is not None:
                _check_grasp_motion(self)
            if self.inserting:
                _check_insertion_motion(self, self.task, self.target_name)
            if self.expert_success_monitor is not None:
                self.expert_success_monitor.update()
            return result

    with ExitStack() as resources:
        if session is None or session.env is None:
            cfg = make_tabletop_environment_config(seed, task)
            env = RecordingEnv(cfg=cfg)
            env.task, env.target_name = task, target_name
            if adaptive_motion:
                env.compact_trajectory = True
                env.motion_tracking_wait_s = 1.0
                env.motion_tracking_tolerance_rad = 0.005
                if not gsmini_base0:
                    env.motion_free_settle_s = 0.0
                    env.motion_insertion_tracking_tolerance_rad = 0.05 if task_id == 48 else 0.10
            initialize_task_environment(session.resources if session is not None else resources,
                                       env, seed, configure_shutdown, task=task)
            if session is not None:
                if isinstance(session, GSminiCollectionSession):
                    session.bind(env)
                else:
                    session.env = env
        else:
            env = session.env
            env.task, env.target_name = task, target_name
            if isinstance(session, GSminiCollectionSession):
                session.reset(short_suite, public_task_id, seed)
            else:
                env.episode = None
                env.grasp_reference = None
                env.inserting = False
                env.expert_success_monitor = None
                env.common_step_counter = env._sim_step_counter = 0
                env.reset(seed=seed)
        if task_id == 51:
            env.insertion_motion_xy_threshold = max(
                goal["xy_threshold"], 0.003)
        elif task_id == 52:
            # Allow brief rim contact while each cuRobo threading waypoint is corrected;
            # final insertion still uses the task's stricter 3 mm tolerance.
            env.insertion_motion_xy_threshold = max(
                goal["xy_threshold"], goal.get("thread_engage_xy_threshold", 0.004) + 0.004)
        scene = env.scene

        def report_object_motion(phase):
            velocity = scene["object"].data.root_vel_w[0]
            linear = torch.linalg.vector_norm(velocity[:3]).item()
            angular = torch.linalg.vector_norm(velocity[3:]).item()
            print(f"[OBJECT] {phase}: linear={linear:.4f}m/s, "
                  f"angular={angular:.4f}rad/s", flush=True)

        def report_bulb_grasp(phase):
            if task_id == 52:
                robot = scene["robot"]
                finger_ids, _ = robot.find_joints(env.cfg.actions.gripper_action.joint_names)
                print(f"[GRASP] {phase}: tcp={scene['ee_frame'].data.target_pos_w[:, 0].tolist()}, "
                      f"object={scene['object'].data.root_pos_w.tolist()}, "
                      f"quat={scene['object'].data.root_quat_w.tolist()}, "
                      f"finger_q={robot.data.joint_pos[:, finger_ids].tolist()}, "
                      f"opening={_gripper_opening(env).tolist()}, "
                      f"velocity={scene['object'].data.root_vel_w.tolist()}", flush=True)

        action = reset_tabletop_task(env, task, suite=TABLETOP_IDENTITIES[tabletop_variant(task)][0], seed=seed,
                                     allow_in_hand=in_hand, joint_defaults=tabletop_joint_defaults)
        if task_id == 52:
            _, target_max, _ = _asset_geometry(task["objects"][target_name]["type"])
            target = env.scene[target_name].data
            rim_local = torch.tensor(
                [[0.0, 0.0, target_max[2] * task["objects"][target_name]["scale"][2]]],
                device=env.device, dtype=target.root_pos_w.dtype,
            ).expand_as(target.root_pos_w)
            env.thread_rim_world = Pose(
                target.root_pos_w, target.root_quat_w,
            ).transform_points(rim_local)
            env.thread_axis_world = _rotation_pose(target.root_quat_w).transform_points(
                torch.tensor([[0.0, 0.0, 1.0]], device=env.device,
                             dtype=target.root_pos_w.dtype).expand_as(target.root_pos_w)
            )
        initial_quaternion = None
        if in_hand:
            env.sim.forward()
            env.scene.update(0.0)
            initial_quaternion = _initialize_in_hand(env, task)
            action[:, -1] = -1.0
            hold(env, app, action, 0.5)
            env.grasp_reference = _tcp_object_pose(env)
        else:
            action[:, -1] = 1.0
            settle_tabletop_scene(env, app, action, task)
        source = scene["object"].data.root_pos_w.clone()
        if task_id == 52:
            initial_quaternion = scene["object"].data.root_quat_w.clone()
        print(f"[RESET] {suite}:{public_task_id} {task['task_name']}; "
              f"object={source.tolist()}, target_root="
              f"{scene[target_name].data.root_pos_w.tolist()}", flush=True)
        report_bulb_grasp("after reset settle")
        world_objects = (("Table",) if not in_hand else ()) + obstacles
        if world_objects:
            world = read_collision_world(env, world_objects)
        else:
            from curobo.scene import Scene

            world = Scene()
        payload_world = read_collision_world(env, (*world_objects, object_name))
        if in_hand:
            full_payload_world = read_collision_world(
                env, _retreat_collision_objects(obstacles, target_name, object_name))
            payload_worlds = (payload_world, full_payload_world)
        else:
            payload_worlds = (payload_world,)
        planner_world = payload_world if task_id == 52 and not in_hand else world
        planner_options = {"velocity_scale": _planner_velocity_scale(task_id),
                           "num_trajopt_seeds": 1 if gsmini_base0 else None}
        if session is not None:
            planner = session.acquire_planner(env, seed, planner_world, *payload_worlds,
                                              **planner_options)
        else:
            planner = resources.enter_context(_create_motion_planner(
                env, seed, planner_world, *payload_worlds, **planner_options))
            planner.warmup(enable_graph=True, num_warmup_iterations=1)

        staged = _start_episode(resources, env, output, suite, public_task_id, seed,
                                task=task,
                                extra_env_args={
                                    "search_policy": "independent seeded physical-slot permutation",
                                    "expert_target_source": "post-contact task metadata teacher; not tactile-only recognition",
                                } if task_id == 50 else (
                                    {"planner_seed": session.planner_seed,
                                     "collection_session": True} if isinstance(session, TubeCollectionSession) else {}))
        grasp_distance_limit = 0.045
        if task_id == 50:
            source_height = _search_sorting_ball(resources, env, app, planner, action, task, seed)
            verify_grasp(env, app, action, source_height, min_height=0.1,
                         max_distance=0.04, min_opening=0.001,
                         object_center_local=env.grasp_center_local,
                         seconds=env.step_dt if adaptive_motion else 1.0)
        elif not in_hand:
            grasp_position = _geometric_grasp_position(env, task)
            print(f"[GRASP] physical bounds center={grasp_position.tolist()}", flush=True)
            grasp_quaternion = None
            if expert_family == "key" or task_id == 48:
                grasp_quaternion = (0.0, 2**-0.5, 2**-0.5, 0.0)
            if task_id == 49:
                grasp_quaternion = (0.0, math.cos(math.pi / 8), math.sin(math.pi / 8), 0.0)
            if task_id == 52:
                grasp_quaternion = _bulb_grasp_quaternion(
                    scene["object"].data.root_quat_w)[0].tolist()
                print(f"[GRASP] bulb-aligned downward orientation={grasp_quaternion}", flush=True)
                overhead = grasp_position.clone()
                overhead[:, 2] += 0.20
                _move(env, app, planner, action, overhead, "grasp_overhead",
                      quaternion=grasp_quaternion)
                _verify_uncontacted_object(env, source, initial_quaternion,
                                           "Grasp overhead move")
                update_collision_world(planner, world)
            if task_id == 48:
                _grasp_horizontal_peg(env, app, planner, action, grasp_position, grasp_quaternion)
                grasp_gap = torch.linalg.vector_norm(env.grasp_reference.position, dim=-1).max().item()
                grasp_distance_limit = max(0.045, grasp_gap + 0.015)
            else:
                grasp = plan_grasp(
                    env, planner, grasp_position=grasp_position,
                    lift_offset=-0.12,
                    # The key head is wider across X than the closed GelSight pad gap.
                    grasp_quaternion=grasp_quaternion,
                    approach_offset=-0.20 if task_id == 52 else -0.10,
                )
                execute_trajectory(env, app, grasp, "approach", action,
                                   settle_seconds=0.0 if adaptive_motion else 0.3,
                                   tracking_wait_s=1.0 if adaptive_motion else 0.0,
                                   max_tracking_error=0.005 if adaptive_motion and not gsmini_base0 else 0.05)
                if task_id == 52:
                    _verify_uncontacted_object(env, source, initial_quaternion,
                                               "Grasp approach")
                report_bulb_grasp("after approach")
                # Thin connectors must reach the grasp height before the fingers close.
                execute_trajectory(env, app, grasp, "grasp", action,
                                   settle_seconds=0.0 if adaptive_motion else 0.5,
                                   tracking_wait_s=1.0 if adaptive_motion else 0.0,
                                   max_tracking_error=(0.0005 if expert_family == "hdmi" else 0.005)
                                   if adaptive_motion and not gsmini_base0 else 0.05)
                if task_id == 52:
                    _verify_uncontacted_object(env, source, initial_quaternion,
                                               "Open-finger descent")
                report_bulb_grasp("after grasp pose")
                if gsmini_base0:
                    close_steps = round(0.35 / env.step_dt)
                    for index in range(1, close_steps + 1):
                        action[:, -1] = 1.0 - 2.0 * index / close_steps
                        step_env(env, app, action)
                    hold(env, app, action, 0.15)
                else:
                    action[:, -1] = -1.0
                    hold(env, app, action, 0.5)
                report_bulb_grasp("after closure")
                env.grasp_reference = _tcp_object_pose(env)
                grasp_gap = torch.linalg.vector_norm(env.grasp_reference.position, dim=-1).max().item()
                if expert_family != "hdmi":
                    grasp_distance_limit = max(0.045, grasp_gap + 0.015)
                execute_trajectory(env, app, grasp, "lift", action,
                                   settle_seconds=0.0 if adaptive_motion else 0.2,
                                   tracking_wait_s=1.0 if adaptive_motion else 0.0,
                                   max_tracking_error=0.005 if adaptive_motion and not gsmini_base0 else 0.05,
                                   repeat_factor=2 if task_id == 52 else 1)
            verify_grasp(env, app, action,
                         source[:, 2],
                         min_height=0.06, max_distance=grasp_distance_limit,
                         min_opening=0.001,
                         object_center_local=None,
                         seconds=env.step_dt if adaptive_motion and task_id != 48 else 1.0)
        if task_id in (49, 50):
            if task_id == 49:
                target = scene[target_name].data.root_pos_w
                low, high = goal["distance_range"]
                current_distance = torch.linalg.vector_norm(
                    scene["object"].data.root_pos_w - target, dim=-1).item()
                if not low <= current_distance <= high:
                    direction = source[:, :2] - target[:, :2]
                    direction /= torch.linalg.vector_norm(
                        direction, dim=-1, keepdim=True,
                    ).clamp_min(1e-6)
                    destination = scene["object"].data.root_pos_w.clone()
                    destination[:, :2] = target[:, :2] + direction * 0.16
                    destination[:, 2] = max(destination[0, 2].item(), source[0, 2].item() + 0.12)
                    _attach_payload_for_planning(env, planner, obstacles, read_collision_world,
                                                 object_name)
                    _move_object_pose(env, app, planner, action,
                                      Pose(destination, scene["object"].data.root_quat_w.clone()),
                                      "retrieval")
                    verify_grasp(
                        env, app, action, source[:, 2], min_height=0.06,
                        max_distance=grasp_distance_limit, min_opening=0.001,
                    )
            verify_task_success(
                env, app, action, task, suite=TABLETOP_IDENTITIES[tabletop_variant(task)][0], backend="gsmini",
                source_height=source_height if task_id == 50 else None,
                object_center_local=env.grasp_center_local if task_id == 50 else None)
            _finish_episode(env, output, staged)
            print(f"[SUCCESS] {suite}:{public_task_id} held goal; saved {output}", flush=True)
            return
        if task_id == 48:
            _reorient_peg(env, app, action)
        _attach_payload_for_planning(
            env, planner, (*obstacles, target_name) if gsmini_threading else obstacles,
            read_collision_world, object_name,
            include_table=not in_hand or gsmini_threading,
        )

        shortcut_transfer = False
        if task_id == 51 or expert_family == "hdmi":
            from benchmark.backends.gsmini.environment import PlanningFailed

            planner.attachment_manager.detach()
            _attach_payload_for_planning(
                env, planner, (*obstacles, target_name), read_collision_world,
                object_name, include_table=True)
            direct_goal = _task_object_goal(env, task, target_name)
            direct_position = direct_goal.position.clone()
            direct_position[:, 2] += 0.05
            try:
                _move_object_pose(env, app, planner, action,
                                  Pose(direct_position, direct_goal.quaternion), "transfer",
                                  allow_gripper_contact=expert_family != "hdmi")
            except PlanningFailed:
                print("[TRANSFER] Direct approach blocked; retaining the staged route", flush=True)
            else:
                shortcut_transfer = True

        if in_hand and not gsmini_threading and not shortcut_transfer:
            object_data = scene["object"].data
            clearance = object_data.root_pos_w.clone()
            clearance[:, 2] += 0.08
            _move_object_pose(
                env, app, planner, action,
                Pose(clearance, object_data.root_quat_w.clone()), "reset_clearance",
            )
            planner.attachment_manager.detach()
            _attach_payload_for_planning(
                env, planner, obstacles, read_collision_world, object_name,
                include_table=True,
            )

        object_goal = _task_object_goal(env, task, target_name)
        if gsmini_base0:
            transfer, _ = _bore_staging_pose(env, goal, object_goal, target_name)
        else:
            transfer_quaternion = (object_goal.quaternion
                                   if (task_id in (48, 52) or
                                       ("expected_object_local_quat" in goal and not in_hand
                                        and not angled_insertion))
                                   else scene["object"].data.root_quat_w)
            above = object_goal.position.clone()
            above[:, 2] += (goal["thread_engagement_range"][1] if gsmini_threading else
                            0.15 if task_id == 52 else (0.05 if in_hand or shortcut_transfer else 0.10))
            transfer = Pose(above, transfer_quaternion)
        if shortcut_transfer:
            transfer = Pose(transfer.position, object_goal.quaternion)
        else:
            _move_object_pose(env, app, planner, action, transfer, "transfer")
        if not angled_insertion:
            stage_tolerance = min(0.003, 0.5 * goal["xy_threshold"])
            for attempt in range(3):
                stage_error = scene["object"].data.root_pos_w - transfer.position
                stage_radial = torch.linalg.vector_norm(stage_error[:, :2], dim=-1).max().item()
                stage_angle = Pose(
                    quaternion=scene["object"].data.root_quat_w).angular_distance(
                        Pose(quaternion=transfer.quaternion)).max().item()
                print(f"[ALIGN] stage radial={stage_radial:.4f}m, "
                      f"angle={stage_angle:.3f}rad", flush=True)
                if stage_radial <= stage_tolerance and stage_angle <= 0.03:
                    break
                if attempt == 2:
                    raise RuntimeError("Could not align the held object above its target")
                _move_object_pose(env, app, planner, action, transfer, "target_align")
        if task_id == 52:
            # Align to the authored socket axis, rather than preserving transfer tilt.
            initial_quaternion = object_goal.quaternion.clone()

        if in_hand:
            planner.attachment_manager.detach()
            _attach_payload_for_planning(
                env, planner, obstacles, read_collision_world, object_name,
                excluded_obstacle_names=(target_name,), include_table=True,
            )
            if "expected_object_local_quat" in goal and not shortcut_transfer:
                _move_object_pose(
                    env, app, planner, action,
                    Pose(transfer.position, object_goal.quaternion), "target_align",
                    repeat_factor=1,
                )
                for attempt in range(3):
                    current = scene["object"].data.root_pos_w
                    radial = torch.linalg.vector_norm(
                        current[:, :2] - transfer.position[:, :2], dim=-1).max().item()
                    print(f"[ALIGN] after rotation radial={radial:.4f}m", flush=True)
                    if radial <= min(0.003, goal["xy_threshold"]):
                        break
                    if attempt == 2:
                        raise RuntimeError("Could not recenter the held object after rotation")
                    _move_object_pose(
                        env, app, planner, action,
                        Pose(transfer.position, object_goal.quaternion), "target_align",
                        repeat_factor=3,
                    )
        else:
            planner.attachment_manager.detach()
            _attach_payload_for_planning(
                env, planner, obstacles, read_collision_world, object_name,
                excluded_obstacle_names=(target_name,),
            )
        env.inserting = True
        insertion_label = "angled_insertion" if angled_insertion else "insertion"
        insertion_options = {}
        if adaptive_insertion:
            insertion_options["settle_seconds"] = 0.0
        try:
            if angled_insertion:
                bore_stage_pose, axis = _bore_staging_pose(env, goal, object_goal, target_name)
                env.inserting = False
                if not gsmini_base0:
                    _move_object_pose(env, app, planner, action, bore_stage_pose, "bore_align",
                                      **insertion_options)
                for attempt in range(3):
                    stage_error = (scene["object"].data.root_pos_w - bore_stage_pose.position)[0]
                    stage_axial = torch.dot(stage_error, axis[0])
                    stage_radial = torch.linalg.vector_norm(
                        stage_error - stage_axial * axis[0]).item()
                    stage_angle = Pose(
                        quaternion=scene["object"].data.root_quat_w).angular_distance(
                            Pose(quaternion=bore_stage_pose.quaternion)).item()
                    print(f"[BORE] staged radial={stage_radial:.4f}m, "
                          f"angle={stage_angle:.3f}rad", flush=True)
                    if stage_radial <= 0.003 and stage_angle <= 0.03:
                        break
                    if attempt == 2:
                        raise RuntimeError("Could not align the held tube above the inclined bore")
                    _move_object_pose(env, app, planner, action, bore_stage_pose, "bore_align",
                                      **insertion_options)
                planner.attachment_manager.detach()
                _attach_payload_for_planning(
                    env, planner, obstacles, read_collision_world, object_name,
                    excluded_obstacle_names=(target_name,),
                )
                env.inserting = True
            elif task_id == 52:
                engage = object_goal.position.clone()
                engage[:, 2] += goal["thread_engagement_range"][1]
                engage_pose = Pose(engage, initial_quaternion)
                for attempt in range(3):
                    if not gsmini_threading or attempt:
                        _move_object_pose(env, app, planner, action, engage_pose, "insertion")
                    actual = Pose(scene["object"].data.root_pos_w,
                                  scene["object"].data.root_quat_w)
                    radial, axial = _insertion_errors(
                        goal, engage_pose,
                        actual.position[0] - engage_pose.position[0])
                    angle = actual.angular_distance(engage_pose).max().item()
                    if radial.item() <= 0.001 and axial.item() <= 0.001 and angle <= 0.03:
                        break
                    if attempt == 2:
                        raise RuntimeError(
                            f"Bulb did not align above socket rim: radial={radial.item():.4f}m, "
                            f"axial={axial.item():.4f}m, angle={angle:.3f}rad")
                    print(f"[THREAD] correcting rim approach {attempt + 1}: "
                          f"radial={radial.item():.4f}m, axial={axial.item():.4f}m, "
                          f"angle={angle:.3f}rad", flush=True)
            if gsmini_base0:
                env.insertion_path_scale = 10.0
                insertion_options["settle_seconds"] = env.step_dt
                env.expert_success_monitor = TaskSuccessMonitor(
                    env, task, suite=TABLETOP_IDENTITIES[tabletop_variant(task)][0], backend="gsmini",
                    initial_quaternion=initial_quaternion)
            if task_id == 52:
                from benchmark.tasks.dynamic.gsmini import _regrasp_threaded_bulb

                def future_thread_goal(duration):
                    return Pose(object_goal.position.clone(), initial_quaternion.clone())

                def regrasp_thread(angle):
                    env.inserting = False
                    try:
                        _regrasp_threaded_bulb(
                            env, app, action, future_thread_goal, angle,
                            lift_speed=0.3, rewind_speed=3.5, max_joint_step=0.3,
                            rewind_max_joint_step=0.4,
                            joint_margin=0.05,
                            reuse_stability=True, gripper_speed=10.0,
                            radial_limit=goal["xy_threshold"],
                            tilt_limit=math.acos(goal["axis_alignment_threshold"]),
                        )
                    finally:
                        env.inserting = True

                def threaded_success():
                    reached = _threaded_seated(env, task, initial_quaternion)
                    if reached:
                        _hold_current_arm_action(env, action)
                    return reached

                thread_options = {"goal_for_duration": future_thread_goal,
                                  "regrasp": regrasp_thread,
                                  "tracking_horizon": 0.2,
                                  "engaged_angular_speed": 2.5,
                                  "rotation_lead_limit": 0.45,
                                  "engaged_max_joint_step": 0.3,
                                  "initial_max_joint_step": 0.3,
                                  "contact_progress": True,
                                  "success_check": threaded_success}
                _execute_threaded_insertion(
                    env, app, planner, action, engage_pose, object_goal.position,
                    initial_quaternion,
                    angular_speed=2.5,
                    **thread_options,
                )
            elif task_id in (48, 51) or angled_insertion:
                # Replan in short moves so contact cannot push the held object
                # sideways between the entry and seated pose.
                start = scene["object"].data.root_pos_w.clone()
                distance = torch.linalg.vector_norm(
                    object_goal.position - start, dim=-1).max().item()
                segment_length = 0.01 if task_id == 51 else 0.04 if task_id == 48 else 0.02
                segments = max(1, math.ceil(distance / segment_length))
                segment_options = dict(insertion_options, repeat_factor=(
                    2 if not angled_insertion and task_id == 12 else
                    insertion_options.get("repeat_factor", 1)))
                for index in range(1, segments + 1):
                    pose = Pose(start + (object_goal.position - start) * (index / segments),
                                object_goal.quaternion)
                    if task_id in (48, 51):
                        segment_options["settle_seconds"] = (
                            0.2 if task_id == 51 and index < segments else 0.0)
                    _move_object_pose(env, app, planner, action, pose, insertion_label,
                                      **segment_options)
            else:
                _move_object_pose(env, app, planner, action, object_goal, insertion_label,
                                  **insertion_options)
            if expert_family == "key":
                # The nominal key goal leaves its tip 0.6 mm above the lock floor.
                # Seat it while still supported so it cannot tip during release.
                seated = object_goal.position.clone()
                seated[:, 2] -= 0.0006
                _move_object_pose(env, app, planner, action,
                                  Pose(seated, object_goal.quaternion), insertion_label,
                                  **insertion_options)
            if adaptive_insertion:
                settle_object(env, app, action,
                              max_seconds=1.5 if task_id == 51 else 1.0, require_drop=False)
        finally:
            env.inserting = False
        if task_id in (51, 52):
            if not adaptive_insertion:
                hold(env, app, action, 0.5)
            report_object_motion("held at insertion endpoint")
        if angled_insertion:
            result = verify_task_success(
                env, app, action, task, suite=TABLETOP_IDENTITIES[tabletop_variant(task)][0], backend="gsmini",
                initial_quaternion=initial_quaternion)
            xy, dz = result["xy_error_m"], result["axial_error_m"]
            _finish_episode(env, output, staged)
            print(f"[SUCCESS] {suite}:{public_task_id} physically held in the target; "
                  f"xy={xy:.4f}m dz={dz:.4f}m; saved {output}", flush=True)
            return
        action[:, -1] = 1.0
        env.grasp_reference = None
        if (adaptive_insertion or gsmini_threading
                or (expert_family == 'key')):
            _open_gripper(env, app, action, require_fully_open=False)
            env.expert_success_monitor = TaskSuccessMonitor(
                env, task, suite=TABLETOP_IDENTITIES[tabletop_variant(task)][0], backend="gsmini",
                initial_quaternion=initial_quaternion)
        else:
            hold(env, app, action, 0.5)
        if task_id in (51, 52):
            report_object_motion("released before retreat")
        planner.attachment_manager.detach()
        update_collision_world(planner, read_collision_world(
            env, _retreat_collision_objects(obstacles, target_name, object_name)))
        retreat = scene["ee_frame"].data.target_pos_w[:, 0].clone()
        retreat[:, 2] += 0.10
        _move(env, app, planner, action, retreat, "retreat",
              allow_gripper_contact=False)
        if task_id in (51, 52):
            report_object_motion("after retreat")
        if task_id not in (12, 24):
            settle_object(env, app, action, max_seconds=5.0, require_drop=False)
        result = verify_task_success(
            env, app, action, task, suite=TABLETOP_IDENTITIES[tabletop_variant(task)][0], backend="gsmini",
            initial_quaternion=initial_quaternion)
        xy, dz = result["xy_error_m"], result["axial_error_m"]
        _finish_episode(env, output, staged)
        print(f"[SUCCESS] {suite}:{public_task_id} physical insertion xy={xy:.4f}m "
              f"dz={dz:.4f}m; saved {output}", flush=True)
