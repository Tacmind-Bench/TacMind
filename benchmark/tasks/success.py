"""Authoritative task success checks shared by expert collection and policy rollout.

These checks only read simulated state. Task geometry, robot drives, contact
materials and sensor requirements remain owned by the existing environments.
"""

from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from benchmark.catalog import get_task, is_extra_scene, is_recognition_task, is_tabletop_task, tabletop_variant
from benchmark.backends.gsmini.motion import _rotation_pose

SUCCESS_FORMAT = "tacmind.task-goals"
MAX_INSERTION_TILT_RAD = 0.17
MAX_LINEAR_SPEED_M_S = 0.01
MAX_ANGULAR_SPEED_RAD_S = 0.1
PEG_MAX_AXIAL_ERROR_M = 0.003
BLIND_SUPPORT_TOLERANCE_M = 0.025
BLIND_FLOOR_TOLERANCE_M = 0.003
BLIND_RIM_TOLERANCE_M = 0.015


class PreLiftHeightReference:
    """Observe a confirmed target grasp and freeze its height when lifting starts.

    Inputs are measured target-center/TCP world positions and target-filtered
    normal contact on both pads. ``step`` is the consecutive control-step index,
    not a physics-substep counter. No Expert phase or action is required.
    """

    def __init__(self):
        self.height_m = None
        self.reference_step = self.locked_step = None
        self._last_step = None
        self._contacts = 0
        self._candidate = None

    @staticmethod
    def _rises(current, previous):
        # Permit float32 transform roundoff, without a physical distance gate.
        ulp = float(np.spacing(np.float32(max(abs(current), abs(previous)))))
        return current - previous > 16 * ulp

    def update(self, center_w, tcp_w, target_normal_forces_N, *, step):
        center = np.asarray(center_w, dtype=float)
        tcp = np.asarray(tcp_w, dtype=float)
        forces = np.asarray(target_normal_forces_N, dtype=float)
        if (center.shape != (3,) or tcp.shape != (3,) or forces.shape != (2,)
                or not all(np.isfinite(value).all() for value in (center, tcp, forces))
                or np.any(forces < 0)
                or np.any(np.abs(np.r_[center, tcp]) > np.finfo(np.float32).max)):
            raise ValueError('Pre-lift reference requires finite positions and nonnegative pad forces')
        if (isinstance(step, (bool, np.bool_)) or not isinstance(step, (int, np.integer))
                or step < 0 or self._last_step is not None and step < self._last_step):
            raise ValueError('Pre-lift reference requires an ordered control-step index')
        if step == self._last_step:
            return self.height_m
        if self._last_step is not None and step != self._last_step + 1:
            self._contacts, self._candidate = 0, None
        self._last_step = int(step)
        if self.height_m is not None:
            return self.height_m

        if self._candidate is not None:
            previous_center, previous_tcp, previous_step = self._candidate
            if self._rises(center[2], previous_center) and self._rises(tcp[2], previous_tcp):
                self.height_m = previous_center
                self.reference_step, self.locked_step = previous_step, int(step)
                return self.height_m
        self._contacts = self._contacts + 1 if np.all(forces > 0) else 0
        self._candidate = ((float(center[2]), float(tcp[2]), int(step))
                           if self._contacts >= 2 else None)
        return None


class GoalNotReached(RuntimeError):
    """A valid physical state has not yet met the task's geometric goal."""


def target_is_released(target_normal_forces_N) -> bool:
    """Require zero measured gripper contact on the target, without a force gate."""
    forces = np.asarray(target_normal_forces_N, dtype=float)
    if not forces.size or not np.isfinite(forces).all() or np.any(forces < 0):
        raise ValueError("Target release requires finite nonnegative target contact forces")
    return bool(np.all(forces == 0))


def _gsmini_target_released(env, task: dict, suite: str) -> bool:
    import torch

    if is_recognition_task(task):
        from benchmark.catalog import candidate_names as _candidate_names

        target_index = _candidate_names(task).index(task["obj_of_interest"][0])
    else:
        # Dynamic contact sensors filter the single payload body.
        target_index = 0
    magnitudes = []
    for side in ("left", "right"):
        for part in ("contact", "case_contact"):
            values = env.scene[f"gsmini_{side}_{part}"].data.force_matrix_w
            if (values.ndim != 4 or values.shape[0] != 1 or values.shape[-1] != 3
                    or target_index >= values.shape[2]):
                raise RuntimeError("GSmini release check requires target-filtered contact forces")
            # Separate body/pad norms prevent opposing contact vectors from canceling.
            magnitudes.append(values[0, :, target_index, :].double().norm(dim=-1))
    return target_is_released(torch.cat(magnitudes).detach().cpu().numpy())


def _tabletop_goal(task: dict) -> dict:
    """Reuse the calibrated key goal for variants with the same asset geometry."""
    goal = task["goals"][0]
    if goal["relationship"] in {"retrieved_from", "lifted_from"}:
        return goal
    if "target_local_offset" not in goal:
        goal = deepcopy(goal)
        reference = get_task("tac_precision", 1)
        reference_goal = reference["goals"][0]
        for role in ("ref_obj", "target"):
            spec, calibrated = task["objects"][goal[role]], reference["objects"][reference_goal[role]]
            if (spec["type"], spec["scale"]) != (calibrated["type"], calibrated["scale"]):
                raise ValueError(f"No calibrated insertion goal for tabletop task {task['task_id']}")
        for key in ("target_local_offset", "expected_object_local_quat"):
            goal.setdefault(key, deepcopy(reference_goal[key]))
    return goal


def _resolved_task(task: dict, suite: str, backend: str) -> dict:
    if suite == "tac_dynamic" and task.get("sensor_profile") != backend:
        if backend == "gsmini":
            from benchmark.tasks.dynamic.scene import resolve_gsmini_task

            return resolve_gsmini_task(task)
    return task


def success_contract(task: dict, *, suite: str, backend: str = "gsmini") -> dict:
    """Describe the same task-specific geometry and persistence for every caller."""
    if suite == "tac_extra" and is_extra_scene(task):
        from benchmark.tasks.extra.scenes import extra_scene_success_contract
        return extra_scene_success_contract(task, backend)
    if suite == "tac_extra" and not is_tabletop_task(task):
        from benchmark.tasks.extra.environment import extra_success_contract

        return extra_success_contract(task, backend)
    if suite not in {"tac_precision", "tac_recog", "tac_dynamic", "tac_extra"}:
        raise ValueError(f"Unknown success suite: {suite}")
    if backend != "gsmini":
        raise ValueError(f"Unknown success backend: {backend}")
    task = _resolved_task(task, suite, backend)
    contract = {"format": SUCCESS_FORMAT, "suite": suite,
                "task_id": task["task_id"], "backend": backend,
                "pose_window_s": 0.0, "stable_steps": 1}
    if suite == "tac_dynamic":
        episode = task["episode_config"]
        keys = ("bore_axis_local", "goal_offset_in_fixture_m", "expected_object_local_quat_wxyz",
                "success_radial_tolerance_m", "success_xy_tolerance_m", "success_axial_range_m",
                "success_z_tolerance_m", "success_axis_alignment", "min_thread_yaw_rad",
                "object_family", "release_after_insert")
        contract.update(frame="fixture", stable_steps=int(episode.get("success_stable_steps", 3)),
                        goal={key: deepcopy(episode[key]) for key in keys if key in episode})
        if episode["object_family"] == "key":
            contract["goal"]["require_full_orientation"] = True
        if contract["stable_steps"] < 1:
            raise ValueError("success_stable_steps must be positive")
        return contract
    if is_recognition_task(task):
        keys = ("relationship", "ref_obj", "target", "ref_local_offset", "xy_threshold", "distance_range",
                "require_released")
        contract.update(frame="world", goals=[{key: deepcopy(goal[key]) for key in keys if key in goal}
                                              for goal in task["goals"]], pose_window_s=0.5,
                        max_position_drift_m=0.005, max_orientation_drift_rad=0.10,
                        support_tolerance_m=BLIND_SUPPORT_TOLERANCE_M * task.get("_eval_object_scale", 1.0),
                        floor_tolerance_m=BLIND_FLOOR_TOLERANCE_M * task.get("_eval_object_scale", 1.0), rim_tolerance_m=BLIND_RIM_TOLERANCE_M * task.get("_eval_object_scale", 1.0))
        return contract
    goal = _tabletop_goal(task)
    relation = goal["relationship"]
    if relation in {"retrieved_from", "lifted_from"}:
        contract.update(frame="world", goal=deepcopy(goal))
        return contract
    family = task.get("episode_config", {}).get("expert_family")
    tight_window = tabletop_variant(task) in (48, 51, 52)
    pose_window = "insertion_entry_local" in goal or family in {"key", "hdmi"} or tight_window
    # Legacy force/contact annotations are not checks in the tabletop expert.
    # In particular, physical Panda grasps do not require GSmini gel contact.
    geometry_keys = ("relationship", "ref_obj", "target", "target_local_offset", "xy_threshold",
                     "height_threshold", "expected_object_local_quat", "insertion_entry_local",
                     "min_thread_yaw_rad", "alignment_axis", "axis_alignment_threshold")
    contract.update(frame="target", goal={key: deepcopy(goal[key]) for key in geometry_keys if key in goal},
                    check_velocity=not pose_window, min_hand_gap_m=0.06,
                    max_tilt_rad=MAX_INSERTION_TILT_RAD, max_linear_speed_m_s=MAX_LINEAR_SPEED_M_S,
                    max_angular_speed_rad_s=MAX_ANGULAR_SPEED_RAD_S,
                    require_two_pad_contact=False)
    if tabletop_variant(task) == 48:
        contract["goal"]["height_threshold"] = min(goal["height_threshold"], PEG_MAX_AXIAL_ERROR_M * task.get("_eval_object_scale", 1.0))
    if pose_window:
        contract.update(pose_window_s=0.5, max_position_drift_m=0.002 if tight_window else 0.005,
                        max_orientation_drift_rad=0.02 if tight_window else 0.10)
    return contract


def _task_object_goal(env, task: dict, target_name: str):
    import torch
    from curobo.types import Pose

    target = env.scene[target_name].data
    device = getattr(env, "device", target.root_pos_w.device)
    goal = _tabletop_goal(task)
    offset = goal["target_local_offset"]
    quaternion = goal.get("expected_object_local_quat", (1.0, 0.0, 0.0, 0.0))
    local = Pose(torch.tensor([offset], device=device, dtype=target.root_pos_w.dtype),
                 torch.tensor([quaternion], device=device,
                              dtype=target.root_quat_w.dtype))
    return Pose(target.root_pos_w, target.root_quat_w).multiply(local)


def _insertion_errors(goal: dict, desired_pose, displacement):
    """Measure radial and axial insertion error in the target's bore frame."""
    import torch

    if "insertion_entry_local" in goal:
        axis_local = torch.zeros_like(desired_pose.quaternion[..., :3])
        axis_local[..., 2] = 1.0
        axis = _rotation_pose(desired_pose.quaternion).transform_points(axis_local)[0]
        axial = torch.dot(displacement, axis)
        radial = torch.linalg.vector_norm(displacement - axial * axis)
        return radial, axial.abs()
    return torch.linalg.vector_norm(displacement[:2]), displacement[2].abs()


def _verify_insertion(env, task: dict, target_name: str, initial_quaternion=None,
                      check_velocity: bool = True, min_hand_gap: float = 0.06):
    import torch
    from curobo.types import Pose

    connector = env.scene["object"].data
    target = env.scene[target_name].data
    tcp = env.scene["ee_frame"].data.target_pos_w
    if not all(torch.isfinite(value).all() for value in
               (connector.root_pos_w, connector.root_quat_w, connector.root_vel_w,
                target.root_pos_w, target.root_quat_w, tcp)):
        raise RuntimeError("Insertion unverified: non-finite scene state")
    if any((torch.linalg.vector_norm(q, dim=-1) <= 1e-7).any()
           for q in (connector.root_quat_w, target.root_quat_w)):
        raise RuntimeError("Insertion unverified: zero quaternion")
    goal = _tabletop_goal(task)
    if goal.get("min_thread_yaw_rad") and (
            initial_quaternion is None or not torch.isfinite(initial_quaternion).all()
            or (torch.linalg.vector_norm(initial_quaternion, dim=-1) <= 1e-7).any()):
        raise RuntimeError("Insertion unverified: invalid thread rotation reference")
    desired = _task_object_goal(env, task, target_name)
    displacement = connector.root_pos_w[0] - desired.position[0]
    radial_error, axial_error = _insertion_errors(goal, desired, displacement)
    xy, dz = radial_error.item(), axial_error.item()
    height_threshold = min(goal["height_threshold"], PEG_MAX_AXIAL_ERROR_M * task.get("_eval_object_scale", 1.0)) if tabletop_variant(task) == 48 else goal["height_threshold"]
    speed = torch.linalg.vector_norm(connector.root_vel_w[0, :3]).item()
    angular_speed = torch.linalg.vector_norm(connector.root_vel_w[0, 3:]).item()
    if "expected_object_local_quat" in goal:
        local_quaternion = torch.tensor([goal["expected_object_local_quat"]],
                                        device=target.root_quat_w.device,
                                        dtype=target.root_quat_w.dtype)
        expected = _rotation_pose(target.root_quat_w).multiply(_rotation_pose(local_quaternion))
    elif goal.get("min_thread_yaw_rad"):
        relative = _rotation_pose(initial_quaternion).inverse().multiply(
            _rotation_pose(connector.root_quat_w))
        signed_twist = 2 * torch.atan2(relative.quaternion[0, 3], relative.quaternion[0, 0])
        signed_twist = torch.remainder(signed_twist + math.pi, 2 * math.pi) - math.pi
        axis = torch.tensor(goal["alignment_axis"], device=connector.root_quat_w.device,
                            dtype=connector.root_quat_w.dtype)[None]
        origin = torch.zeros_like(connector.root_pos_w)
        connector_axis = Pose(origin, connector.root_quat_w).transform_points(axis)
        target_axis = Pose(
            torch.zeros_like(target.root_pos_w), target.root_quat_w,
        ).transform_points(axis)
        aligned = torch.dot(connector_axis[0], target_axis[0]).item()
        if (aligned < goal["axis_alignment_threshold"]
                or signed_twist > -goal["min_thread_yaw_rad"]):
            raise GoalNotReached(
                f"Insertion unverified: thread yaw={signed_twist.item():.3f}rad, "
                f"axis={aligned:.3f}")
        expected = Pose(quaternion=connector.root_quat_w)
    else:
        expected = Pose(quaternion=target.root_quat_w)
    tilt = Pose(quaternion=connector.root_quat_w).angular_distance(expected).item()
    hand_gap = torch.linalg.vector_norm(tcp[0, 0] - connector.root_pos_w[0]).item()
    if (not all(math.isfinite(value) for value in (xy, dz, speed, angular_speed, tilt, hand_gap))
            or xy >= goal["xy_threshold"] or dz >= height_threshold
            or (check_velocity and (speed >= MAX_LINEAR_SPEED_M_S or angular_speed >= MAX_ANGULAR_SPEED_RAD_S))
            or tilt >= MAX_INSERTION_TILT_RAD or hand_gap <= min_hand_gap):
        joint_velocity = getattr(env.scene["robot"].data, "joint_vel",
                                 torch.zeros_like(env.scene["robot"].data.joint_pos))
        raise GoalNotReached(
            f"Insertion unverified: xy={xy:.4f}m, dz={dz:.4f}m, "
            f"xyz={displacement.tolist()}, "
            f"tilt={tilt:.3f}rad, speed={speed:.4f}m/s, "
            f"linear={connector.root_vel_w[0, :3].tolist()}, "
            f"angular={connector.root_vel_w[0, 3:].tolist()}, "
            f"joint_speed={torch.linalg.vector_norm(joint_velocity[0]).item():.4f}rad/s, "
            f"hand gap={hand_gap:.3f}m")
    return xy, dz


def _verify_goal(env, task: dict, goal: dict) -> dict:
    import torch
    from benchmark.backends.gsmini.geometry import _object_center_local, _object_bottom_offset, _collision_height
    from curobo.types import Pose

    object_data = env.scene["object"].data
    target_data = env.scene[goal["target"]].data
    if (not all(torch.isfinite(value).all() for value in
                (object_data.root_pos_w, object_data.root_quat_w, target_data.root_pos_w))
            or (torch.linalg.vector_norm(object_data.root_quat_w, dim=-1) <= 1e-7).any()):
        raise RuntimeError(f"Recog goal {goal['relationship']} failed: invalid scene pose")
    center = _object_center_local(env, task, goal.get("ref_local_offset"))
    world_center = Pose(object_data.root_pos_w, object_data.root_quat_w).transform_points(
        center.expand_as(object_data.root_pos_w))
    delta = world_center[0] - target_data.root_pos_w[0]
    distance = torch.linalg.vector_norm(delta[:2]).item()
    relation = goal["relationship"]
    if relation == "retrieved_from":
        distance = torch.linalg.vector_norm(delta).item()
        low, high = goal["distance_range"]
        valid = low <= distance <= high
    elif relation in {"on", "in"}:
        valid = distance <= goal["xy_threshold"]
        if relation == "on":
            support = _collision_height(env, goal["target"], highest=True)
            bottom = object_data.root_pos_w[0, 2] + _object_bottom_offset(env, task)[0]
            valid &= abs(bottom - support) <= BLIND_SUPPORT_TOLERANCE_M * task.get("_eval_object_scale", 1.0)
        else:
            rim = _collision_height(env, goal["target"], highest=True)
            floor = _collision_height(env, goal["target"], highest=False)
            bottom = object_data.root_pos_w[0, 2] + _object_bottom_offset(env, task)[0]
            # Keep the object above the container's outer bottom, with the
            # same 3 mm tolerance used for placement clearance.
            valid &= bottom >= floor - BLIND_FLOOR_TOLERANCE_M * task.get("_eval_object_scale", 1.0)
            valid &= world_center[0, 2].item() <= rim + BLIND_RIM_TOLERANCE_M * task.get("_eval_object_scale", 1.0)
    else:
        raise ValueError(f"Unsupported Recog relationship: {relation}")
    if not valid:
        raise GoalNotReached(
            f"Recog goal {relation} failed: distance={distance:.4f}m, "
            f"target={goal['target']}, "
            f"threshold={goal.get('xy_threshold', goal.get('distance_range'))}")

    for decoy_goal in task["goals"]:
        decoy_relation = decoy_goal["relationship"]
        if decoy_goal is goal:
            continue
        decoy = env.scene[decoy_goal["ref_obj"]].data.root_pos_w[0]
        decoy_target = env.scene[decoy_goal["target"]].data.root_pos_w[0]
        if not torch.isfinite(torch.cat((decoy, decoy_target))).all():
            raise RuntimeError(f"Decoy {decoy_goal['ref_obj']} has invalid scene position")
        separation = torch.linalg.vector_norm(decoy[:2] - decoy_target[:2]).item()
        if decoy_relation == "retrieved_from":
            low, high = decoy_goal["distance_range"]
            if not low <= torch.linalg.vector_norm(decoy - decoy_target).item() <= high:
                raise GoalNotReached(f"Decoy {decoy_goal['ref_obj']} left its source region")
        elif (decoy_relation in {"not_on", "not_in"}
              and separation <= decoy_goal["xy_threshold"]):
            raise GoalNotReached(f"Decoy {decoy_goal['ref_obj']} was placed in the target region")
    return {"success": True, "distance_error_m": distance}


def _dynamic_success(env, task: dict) -> dict[str, float | bool]:
    import torch
    from curobo.types import Pose
    from benchmark.tasks.dynamic.scene import dynamic_goal_quaternion

    episode = task["episode_config"]
    obj = env.scene["object"].data
    fixture = env.scene["fixture"].data
    poses = (obj.root_pos_w, obj.root_quat_w, fixture.root_pos_w, fixture.root_quat_w)
    if (not all(torch.isfinite(value).all() for value in poses)
            or any((torch.linalg.vector_norm(q, dim=-1) <= 1e-7).any()
                   for q in (obj.root_quat_w, fixture.root_quat_w))):
        return {"success": False, "reason": "invalid scene pose"}
    relative = Pose(fixture.root_pos_w, fixture.root_quat_w).inverse().multiply(
        Pose(obj.root_pos_w, obj.root_quat_w))
    axis = torch.as_tensor(episode["bore_axis_local"], device=env.device,
                           dtype=relative.position.dtype)
    axis = axis / torch.linalg.vector_norm(axis).clamp_min(1.0e-8)
    error = relative.position[0] - torch.as_tensor(
        episode["goal_offset_in_fixture_m"], device=env.device,
        dtype=relative.position.dtype,
    )
    axial = torch.dot(error, axis)
    radial_error = error - axial * axis
    radial = torch.linalg.vector_norm(radial_error).item()
    expected_quaternion = torch.as_tensor(
        dynamic_goal_quaternion(task), device=env.device,
        dtype=relative.quaternion.dtype,
    ).expand_as(relative.quaternion)
    local_z = torch.zeros_like(relative.position)
    local_z[:, 2] = 1.0
    expected_axis = Pose(torch.zeros_like(relative.position), expected_quaternion).transform_points(
        local_z)
    actual_axis = Pose(torch.zeros_like(relative.position), relative.quaternion).transform_points(
        local_z)
    alignment = torch.sum(expected_axis * actual_axis, dim=-1).min().item()
    radial_tolerance = float(episode.get(
        "success_radial_tolerance_m", episode.get("success_xy_tolerance_m", 0.004)))
    axial_range = episode.get("success_axial_range_m")
    axial_ok = (float(axial_range[0]) <= axial.item() <= float(axial_range[1])
                if axial_range else abs(axial.item()) <= float(
                    episode.get("success_z_tolerance_m", 0.008)))
    angle_limit = float(episode.get("success_axis_alignment", 0.94))
    orientation_error = None
    orientation_ok = True
    if episode["object_family"] in {"power_plug", "usb", "key", "hdmi"}:
        # Keyed connectors must match the whole authored pose, including twist.
        orientation_error = Pose(quaternion=relative.quaternion).angular_distance(
            Pose(quaternion=expected_quaternion)).max().item()
        orientation_ok = math.cos(orientation_error) >= angle_limit
    thread_yaw = None
    thread_ok = True
    if episode["object_family"] == "bulb" and task.get("sensor_profile") == "gsmini":
        twist = _rotation_pose(expected_quaternion).inverse().multiply(
            _rotation_pose(relative.quaternion)).quaternion[0]
        thread_yaw = float(2 * torch.atan2(twist[3], twist[0]))
        thread_yaw = (thread_yaw + math.pi) % (2 * math.pi) - math.pi
        thread_ok = thread_yaw <= -float(episode["min_thread_yaw_rad"])
    return {
        "success": (radial <= radial_tolerance and axial_ok and alignment >= angle_limit
                    and orientation_ok and thread_ok),
        "radial_error_m": radial,
        "radial_error_local_m": tuple(float(value) for value in radial_error.tolist()),
        "axial_error_m": axial.item(),
        "axis_alignment": alignment,
        **({"orientation_error_rad": orientation_error} if orientation_error is not None else {}),
        **({"thread_yaw_rad": thread_yaw} if thread_yaw is not None else {}),
    }


class PoseStabilityWindow:
    """Observe the collector's pose drift window without advancing simulation."""

    def __init__(self, pose, *, step_dt: float, duration_s: float = 0.5,
                 max_position_drift: float = 0.005, max_orientation_drift: float = 0.10):
        self.steps = max(1, math.ceil(duration_s / step_dt))
        self.max_position_drift = max_position_drift
        self.max_orientation_drift = max_orientation_drift
        self.reset(pose)

    def reset(self, pose):
        self.start = pose.clone()
        self.stable_steps = 0
        self.peak_position_drift = self.peak_orientation_drift = 0.0

    def observe(self, pose, *, advanced: bool = True):
        import torch

        position_drift = torch.linalg.vector_norm(
            pose.position - self.start.position, dim=-1).max().item()
        orientation_drift = pose.angular_distance(self.start).max().item()
        quaternion_valid = (torch.isfinite(pose.quaternion).all()
                            & (torch.linalg.vector_norm(pose.quaternion, dim=-1) > 1e-7).all())
        if (not math.isfinite(position_drift) or not math.isfinite(orientation_drift)
                or not quaternion_valid.item()):
            raise RuntimeError("Pose stability exceeded pose drift limits: non-finite pose or zero quaternion")
        outside = (position_drift > self.max_position_drift
                   or orientation_drift > self.max_orientation_drift)
        if outside:
            self.reset(pose)
        else:
            self.stable_steps += int(advanced)
            self.peak_position_drift = max(self.peak_position_drift, position_drift)
            self.peak_orientation_drift = max(self.peak_orientation_drift, orientation_drift)
        return outside, position_drift, orientation_drift

    @property
    def ready(self) -> bool:
        return self.stable_steps >= self.steps


def check_task_success(env, task: dict, *, suite: str, backend: str = "gsmini",
                       initial_quaternion=None, source_height=None,
                       object_center_local=None, contract: dict | None = None) -> dict:
    """Read geometric success; ordinary goal misses return a reason, invalid state raises."""
    if suite == "tac_extra" and is_extra_scene(task):
        from benchmark.tasks.extra.scenes import ExtraSceneSuccessMonitor
        return ExtraSceneSuccessMonitor(env, task, backend).update(advanced=False)
    if suite == "tac_extra" and not is_tabletop_task(task):
        from benchmark.tasks.extra.environment import ExtraSuccessMonitor

        return ExtraSuccessMonitor(env, task, backend).update(advanced=False)
    import torch
    from curobo.types import Pose

    task = _resolved_task(task, suite, backend)
    contract = contract if contract is not None else success_contract(task, suite=suite, backend=backend)
    try:
        if suite == "tac_dynamic":
            result = _dynamic_success(env, task)
            if result.get("reason") == "invalid scene pose":
                raise RuntimeError("TacDynamic success check received an invalid scene pose")
            if (result["success"] and backend == "gsmini"
                    and task["episode_config"].get("release_after_insert", False)):
                result["target_released"] = _gsmini_target_released(env, task, suite)
                if not result["target_released"]:
                    result.update(success=False, reason="target is still in gripper contact")
            return result
        if is_recognition_task(task):
            goal = next(goal for goal in task["goals"]
                        if goal["ref_obj"] == task["obj_of_interest"][0])
            result = _verify_goal(env, task, goal)
            if backend == "gsmini" and goal.get("require_released", False):
                result["target_released"] = _gsmini_target_released(env, task, suite)
                if not result["target_released"]:
                    result.update(success=False, reason="target is still in gripper contact")
            return result
        goal = _tabletop_goal(task)
        relation = goal["relationship"]
        if relation in {"retrieved_from", "lifted_from"}:
            obj = env.scene["object"].data
            if (not all(torch.isfinite(value).all() for value in (obj.root_pos_w, obj.root_quat_w))
                    or (torch.linalg.vector_norm(obj.root_quat_w, dim=-1) <= 1e-7).any()):
                raise RuntimeError("Held goal received an invalid scene pose")
            if relation == "retrieved_from":
                target = env.scene[goal["target"]].data.root_pos_w
                if not torch.isfinite(target).all():
                    raise RuntimeError("Retrieval target received an invalid scene position")
                value = torch.linalg.vector_norm(obj.root_pos_w - target, dim=-1).item()
                low, high = goal["distance_range"]
                metric = "retrieval_distance_m"
            else:
                if source_height is None:
                    raise ValueError("Lift success requires the pre-lift source height")
                if backend == "gsmini":
                    if object_center_local is None:
                        object_center_local = env.grasp_center_local
                    center = Pose(obj.root_pos_w, obj.root_quat_w).transform_points(object_center_local)
                else:
                    center = obj.root_pos_w
                value = (center[:, 2] - source_height).item()
                low, high = goal["vertical_separation_range"]
                metric = "lift_height_m"
            if not math.isfinite(value):
                raise RuntimeError("Held goal received a non-finite measurement")
            return {"success": low <= value <= high, metric: value,
                    **({"reason": f"{relation}: {value:.4f}m outside [{low}, {high}]"}
                       if not low <= value <= high else {})}
        xy, dz = _verify_insertion(
            env, task, goal["target"], initial_quaternion,
            check_velocity=contract["check_velocity"], min_hand_gap=contract["min_hand_gap_m"])
        if contract["require_two_pad_contact"] and not env.cfg.grasp_geometry(env)[1].all():
            raise GoalNotReached("Held gear lost two-sided gripper contact")
        return {"success": True, "xy_error_m": xy, "axial_error_m": dz}
    except GoalNotReached as exc:
        return {"success": False, "reason": str(exc)}


class TaskSuccessMonitor:
    """Per-episode geometry and persistence, used by both collection and rollout."""

    def __init__(self, env, task: dict, *, suite: str, backend: str = "gsmini",
                 initial_quaternion=None, source_height=None, object_center_local=None):
        if suite == "tac_extra" and is_extra_scene(task):
            from benchmark.tasks.extra.scenes import ExtraSceneSuccessMonitor
            self._extra_scene = ExtraSceneSuccessMonitor(env, task, backend)
            self.contract, self.window = self._extra_scene.contract, None
            return
        if suite == "tac_extra" and not is_tabletop_task(task):
            from benchmark.tasks.extra.environment import ExtraSuccessMonitor

            self._extra = ExtraSuccessMonitor(env, task, backend)
            self.contract, self.window = self._extra.contract, None
            return
        from curobo.types import Pose

        self.env, self.suite, self.backend = env, suite, backend
        self.contract = success_contract(task, suite=suite, backend=backend)
        self.task = _resolved_task(task, suite, backend)
        obj = env.scene["object"].data
        if (initial_quaternion is None and is_tabletop_task(task) and backend == "gsmini"
                and tabletop_variant(self.task) == 52):
            # Collection measures bulb twist from its aligned socket goal.
            initial_quaternion = _task_object_goal(env, self.task, self.task["goals"][0]["target"]).quaternion
        self.initial_quaternion = (obj.root_quat_w.clone() if initial_quaternion is None
                                   else initial_quaternion.clone())
        self.source_height = source_height
        self.object_center_local = object_center_local
        self.lift_reference = None
        if is_tabletop_task(task) and self.task["goals"][0]["relationship"] == "lifted_from":
            if backend == "gsmini" and self.object_center_local is None:
                from benchmark.backends.gsmini.geometry import _object_center_local

                self.object_center_local = _object_center_local(env, self.task)[None]
            if source_height is None:
                if backend == "gsmini" and tabletop_variant(self.task) == 50:
                    from benchmark.catalog import candidate_names as _candidate_names

                    self.lift_reference = PreLiftHeightReference()
                    self._sorting_contact_index = _candidate_names(self.task).index(self.task["obj_of_interest"][0])
                    self._update_lift_reference(getattr(env, "common_step_counter", 0))
                else:
                    center = (Pose(obj.root_pos_w, obj.root_quat_w).transform_points(self.object_center_local)
                              if backend == "gsmini" else obj.root_pos_w)
                    self.source_height = center[:, 2].clone()
        self.window = None
        if self.contract["pose_window_s"]:
            self.window = PoseStabilityWindow(
                Pose(obj.root_pos_w, obj.root_quat_w), step_dt=env.step_dt,
                duration_s=self.contract["pose_window_s"],
                max_position_drift=self.contract["max_position_drift_m"],
                max_orientation_drift=self.contract["max_orientation_drift_rad"])
        self.stable_steps = 0
        self._last_control_step = getattr(env, "common_step_counter", None)

    def _update_lift_reference(self, control_step):
        import torch
        from curobo.types import Pose

        obj = self.env.scene["object"].data
        center = Pose(obj.root_pos_w, obj.root_quat_w).transform_points(self.object_center_local)
        tcp = self.env.scene["ee_frame"].data.target_pos_w[0, 0]
        forces = torch.stack([
            self.env.scene[f"gsmini_{side}_contact"].data.force_matrix_w[
                0, :, self._sorting_contact_index, :].sum(dim=0).norm()
            for side in ("left", "right")])
        height = self.lift_reference.update(center[0].detach().cpu().numpy(),
            tcp.detach().cpu().numpy(), forces.detach().cpu().numpy(), step=control_step)
        if height is not None:
            self.source_height = center.new_tensor([height])

    def update(self, *, advanced: bool = True) -> dict:
        if hasattr(self, "_extra_scene"):
            return self._extra_scene.update(advanced=advanced)
        if hasattr(self, "_extra"):
            return self._extra.update(advanced=advanced)
        from curobo.types import Pose

        control_step = getattr(self.env, "common_step_counter", None)
        if control_step is not None:
            advanced = advanced and control_step != self._last_control_step
            self._last_control_step = control_step
        if self.lift_reference is not None:
            if advanced:
                self._update_lift_reference(control_step)
            if self.source_height is None:
                self.stable_steps = 0
                return {"success": False, "reason": "waiting for the grasped target to begin lifting"}
        result = check_task_success(
            self.env, self.task, suite=self.suite, backend=self.backend,
            initial_quaternion=self.initial_quaternion, source_height=self.source_height,
            object_center_local=self.object_center_local, contract=self.contract)
        if self.lift_reference is not None:
            result["lift_source_height_m"] = self.lift_reference.height_m
        if self.window is not None:
            obj = self.env.scene["object"].data
            pose = Pose(obj.root_pos_w, obj.root_quat_w)
            # Start at the first valid goal sample. Stability before entering
            # the target, or across a goal miss, cannot complete the task.
            if not result["success"] or self.stable_steps == 0:
                self.window.reset(pose)
            else:
                self.window.observe(pose, advanced=advanced)
        if not result["success"]:
            self.stable_steps = 0
        elif advanced or self.stable_steps == 0:
            self.stable_steps += 1
        if result["success"] and (self.stable_steps < self.contract["stable_steps"]
                                  or (self.window is not None and not self.window.ready)):
            return {**result, "success": False, "reason": "task stability window incomplete"}
        if result["success"] and self.window is not None:
            result.update(peak_position_drift_m=self.window.peak_position_drift,
                          peak_orientation_drift_rad=self.window.peak_orientation_drift,
                          stable_duration_s=self.window.stable_steps * self.env.step_dt)
        if (result["success"] and self.suite == "tac_dynamic" and self.backend == "gsmini"
                and self.task["task_id"] == 8):
            from benchmark.tasks.dynamic.scene import _verify_target_insertion

            roles = ("object", "fixture")
            paths = {role: self.env.scene[role].cfg.prim_path for role in roles}
            poses = {role: (self.env.scene[role].data.root_pos_w[0].cpu().tolist(),
                            self.env.scene[role].data.root_quat_w[0].cpu().tolist())
                     for role in roles}
            try:
                result["target_insertion"] = _verify_target_insertion(self.env.sim.stage, paths, poses)
            except GoalNotReached as error:
                return {**result, "success": False, "reason": str(error)}
        return result


def verify_task_success(env, app, action, task: dict, *, suite: str, backend: str = "gsmini",
                        initial_quaternion=None, source_height=None, object_center_local=None,
                        max_wait_seconds: float | None = None) -> dict:
    """Complete the shared goal window, including previously observed insertion steps."""
    from benchmark.backends.gsmini.execution import step_env

    monitor = getattr(env, "expert_success_monitor", None)
    if monitor is None:
        monitor = TaskSuccessMonitor(
            env, task, suite=suite, backend=backend, initial_quaternion=initial_quaternion,
            source_height=source_height, object_center_local=object_center_local)
    required = (monitor.window.steps if monitor.window is not None
                else monitor.contract["stable_steps"] - 1)
    if suite == "tac_extra" and not is_extra_scene(task) and not is_tabletop_task(task):
        required = monitor.contract["stable_steps"]
    if max_wait_seconds is not None:
        if not math.isfinite(max_wait_seconds) or max_wait_seconds < required * env.step_dt:
            raise ValueError("Success wait must cover the task stability window")
        required = math.ceil(max_wait_seconds / env.step_dt)
    result = monitor.update(advanced=False)
    for _ in range(required):
        if result["success"]:
            break
        if suite == "tac_extra" and not is_extra_scene(task) and not is_tabletop_task(task):
            env.step(action)
        else:
            step_env(env, app, action)
        result = monitor.update()
    if not result["success"]:
        raise RuntimeError(f"{suite}:{task['task_id']} success unverified: {result}")
    return result
