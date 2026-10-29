# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared simulation lifecycle, stepping and trajectory execution."""
from contextlib import contextmanager
import math
import traceback
import sys


class SimulationClosed(InterruptedError):
    """The user closed the application or stopped its physics timeline."""


@contextmanager
def simulation_app(launcher, *, require_completion=False):
    """Close the app; require headless runs to finish grasp verification."""
    try:
        try:
            yield launcher.app
        except SimulationClosed:
            if require_completion:
                raise RuntimeError("Simulation closed before grasp verification completed") from None
            print("[EXIT] Simulation closed by user.", flush=True)
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        launcher.app.app.post_quit(1)
        raise
    finally:
        launcher.app.close()


def configure_shutdown(env) -> None:
    """Let the example unwind instead of blocking inside Isaac Lab's STOP callback."""
    # Isaac Lab exposes no public switch for this standalone app-control callback.
    # It loops until PLAY, even when a window close has already requested exit.
    handle = env.sim._app_control_on_stop_handle
    if handle is not None:
        handle.unsubscribe()
        env.sim._app_control_on_stop_handle = None


def check_grasp(height: float, distance: float, opening: float, *,
                min_height: float = 0.15, max_distance: float = 0.035,
                min_opening: float = 0.01) -> None:
    """Reject invalid telemetry as well as dropped, distant, or empty grasps."""
    if (not all(math.isfinite(value) for value in (height, distance, opening))
            or height < min_height or distance > max_distance or opening < min_opening):
        raise RuntimeError(f"Grasp failed: lift={height:.3f}, TCP distance={distance:.3f}, width={opening:.3f}")


def plan_grasp(env, planner, *, grasp_position=None, lift_offset=-0.20,
               grasp_quaternion=None, approach_offset=-0.10):
    """Use the official TCP offset and cuRobo's three-phase grasp planner."""
    from curobo.types import GoalToolPose
    from benchmark.backends.gsmini.motion import _planner_joint_state
    from benchmark.backends.gsmini.workspace import grasp_poses

    robot, obj = env.scene["robot"], env.scene["object"]
    grasp = grasp_poses(env, obj.data.root_pos_w.clone() if grasp_position is None else grasp_position,
                        quaternion=grasp_quaternion)
    current = _planner_joint_state(robot, planner)
    plan = planner.plan_grasp(
        grasp_poses=GoalToolPose.from_poses({"panda_hand": grasp}),
        current_state=current,
        grasp_approach_offset=approach_offset,
        grasp_lift_offset=lift_offset,
        disable_collision_links=[],  # Keep hand/finger checks against the table.
    )
    if plan.success is None or not plan.success.all().item():
        raise RuntimeError(f"Grasp planning failed: {plan.status}; success={plan.success}")
    print(f"[PLAN] {plan.status}", flush=True)
    return plan


def step_env(env, app, action, *, return_termination: bool = False) -> tuple | None:
    """Advance via the action manager; optionally return termination flags for evaluation."""
    # Keep pause responsive without entering SimulationContext.step's pause loop.
    while not env.sim.is_playing():
        if not app.is_running() or env.sim.is_stopped():
            raise SimulationClosed()
        env.sim.render()
    if not app.is_running() or env.sim.is_stopped():
        raise SimulationClosed()
    try:
        result = env.step(action)
        if len(result) == 2:
            import torch
            _, info = result
            terminated = torch.as_tensor(info["terminated"], device=env.device, dtype=torch.bool).reshape(env.num_envs)
            truncated = torch.as_tensor(info["truncated"], device=env.device, dtype=torch.bool).reshape(env.num_envs)
        else:
            _, _, terminated, truncated, _ = result
    except Exception:
        # A GUI STOP during rendering invalidates the physics/sensor views before
        # env.step returns. Unwind those shutdown errors, but retain real failures.
        if not app.is_running() or env.sim.is_stopped():
            raise SimulationClosed() from None
        raise
    if not app.is_running() or env.sim.is_stopped():
        raise SimulationClosed()
    if return_termination:
        return terminated, truncated
    if (terminated | truncated).any():
        raise RuntimeError("Lift environment terminated during execution")


def hold(env, app, action, seconds: float) -> None:
    for _ in range(round(seconds / env.step_dt)):
        step_env(env, app, action)


def settle_scene(env, app, action, task: dict, max_seconds: float = 12.0) -> None:
    """Start recording only after all dynamic task objects settle physically."""
    import math
    import torch
    from curobo.types import Pose
    from benchmark.backends.gsmini.geometry import _asset_geometry

    dynamic = [(name, env.scene["object" if name == task["obj_of_interest"][0] else name])
               for name, spec in task["objects"].items() if spec.get("kinematic") is not True]

    def validate_states():
        for name, asset in dynamic:
            data = asset.data
            if (not all(torch.isfinite(value).all() for value in
                        (data.root_pos_w, data.root_quat_w, data.root_vel_w))
                    or (torch.linalg.vector_norm(data.root_quat_w, dim=-1) <= 1e-7).any()):
                raise RuntimeError(f"TacRecog object {name} has invalid state during settling")

    validate_states()
    sphere_centers = {}
    for name, asset in dynamic:
        spec = task["objects"][name]
        if spec["type"].startswith("ball_class_"):
            minimum, maximum, _ = _asset_geometry(spec["type"])
            sphere_centers[name] = torch.tensor(
                [(low + high) * 0.5 for low, high in zip(minimum, maximum)],
                device=env.device, dtype=asset.data.root_pos_w.dtype,
            ) * torch.tensor(spec["scale"], device=env.device)

    def settling_pose(name, asset):
        pose = Pose(asset.data.root_pos_w.clone(), asset.data.root_quat_w.clone())
        if name in sphere_centers:
            # Root pivots orbit a rotating ball; only its geometric center
            # determines whether a sphere is stationary enough to approach.
            position = pose.transform_points(sphere_centers[name].expand_as(pose.position))
            quaternion = torch.zeros_like(pose.quaternion)
            quaternion[:, 0] = 1.0
            return Pose(position, quaternion)
        return pose

    pose_reference = {name: settling_pose(name, asset) for name, asset in dynamic}
    quiet = 0
    pose_quiet = 0
    # Rolling over a dimple can produce a short pause without coming to rest.
    required = max(1, math.ceil((1.0 if sphere_centers else 0.25) / env.step_dt))
    for _ in range(math.ceil(max_seconds / env.step_dt)):
        step_env(env, app, action)
        validate_states()
        still = not sphere_centers and all(
            torch.linalg.vector_norm(asset.data.root_vel_w[:, :3], dim=-1).max() < 0.01
            and torch.linalg.vector_norm(asset.data.root_vel_w[:, 3:], dim=-1).max() < 0.1
            for _, asset in dynamic
        )
        quiet = quiet + 1 if still else 0
        if quiet >= required:
            print(f"[SETTLE] Dynamic objects stable for {quiet * env.step_dt:.2f}s", flush=True)
            return
        poses = {name: settling_pose(name, asset) for name, asset in dynamic}
        poses_still = all(
            torch.linalg.vector_norm(pose.position - pose_reference[name].position,
                                     dim=-1).max() < 0.001
            and pose.angular_distance(pose_reference[name]).max() < 0.01
            for name, pose in poses.items()
        )
        if poses_still:
            pose_quiet += 1
        else:
            pose_quiet = 0
            pose_reference = poses
        if pose_quiet >= required:
            print(f"[SETTLE] Dynamic object poses stable for "
                  f"{pose_quiet * env.step_dt:.2f}s despite velocity jitter", flush=True)
            return
    moving = {
        name: {
            "linear": round(torch.linalg.vector_norm(
                asset.data.root_vel_w[:, :3], dim=-1).max().item(), 4),
            "angular": round(torch.linalg.vector_norm(
                asset.data.root_vel_w[:, 3:], dim=-1).max().item(), 4),
            "position": tuple(round(value, 4) for value in asset.data.root_pos_w[0].tolist()),
        }
        for name, asset in dynamic
        if (torch.linalg.vector_norm(asset.data.root_vel_w[:, :3], dim=-1).max() >= 0.01
            or torch.linalg.vector_norm(asset.data.root_vel_w[:, 3:], dim=-1).max() >= 0.1)
    }
    raise RuntimeError(f"TacRecog objects did not settle within {max_seconds:g}s: {moving}")



def settle_object(env, app, action, max_seconds: float = 1.5, *,
                  max_displacement: float = 0.03, require_drop: bool = True,
                  initial_position=None, max_angular_speed: float = 0.1,
                  require_velocity_quiet: bool = False, object_center_local=None) -> None:
    """Wait for settling; initial_position uses the supplied center, or root."""
    import torch
    from curobo.types import Pose

    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be positive and finite")
    if not math.isfinite(max_displacement) or max_displacement <= 0:
        raise ValueError("max_displacement must be positive and finite")
    if not math.isfinite(max_angular_speed) or max_angular_speed <= 0:
        raise ValueError("max_angular_speed must be positive and finite")
    data = env.scene["object"].data
    center_local = (None if object_center_local is None else torch.as_tensor(
        object_center_local, device=env.device, dtype=data.root_pos_w.dtype,
    ).expand_as(data.root_pos_w))

    def measured_pose():
        pose = Pose(data.root_pos_w.clone(), data.root_quat_w.clone())
        if center_local is not None:
            return Pose(pose.transform_points(center_local), pose.quaternion)
        return pose

    pose_reference = measured_pose()
    initial = (pose_reference.position if initial_position is None else initial_position).clone()
    quiet = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    pose_quiet = torch.zeros_like(quiet)
    quiet_steps = max(1, math.ceil(0.1 / env.step_dt))
    min_steps = max(1, math.ceil(0.15 / env.step_dt))
    for step in range(math.ceil(max_seconds / env.step_dt)):
        step_env(env, app, action)
        pose = measured_pose()
        velocity = data.root_vel_w
        quaternion = data.root_quat_w
        if (not torch.isfinite(quaternion).all()
                or (torch.linalg.vector_norm(quaternion, dim=-1) <= 0).any()):
            raise RuntimeError(f"Invalid object orientation during settling: {quaternion.tolist()}")
        displacement = torch.linalg.vector_norm(pose.position - initial, dim=-1)
        if (not torch.isfinite(pose.position).all() or not torch.isfinite(velocity).all()
                or (displacement > max_displacement).any()):
            linear_speed = torch.linalg.vector_norm(velocity[:, :3], dim=-1).max().item()
            angular_speed = torch.linalg.vector_norm(velocity[:, 3:], dim=-1).max().item()
            raise RuntimeError(
                f"Reset object left the tabletop during settling: "
                f"drift={displacement.max().item():.4f}m, "
                f"position={pose.position.tolist()}, linear={linear_speed:.3f}m/s, "
                f"angular={angular_speed:.3f}rad/s")
        stationary = (torch.linalg.vector_norm(velocity[:, :3], dim=-1) < 0.01) & (
            torch.linalg.vector_norm(velocity[:, 3:], dim=-1) < max_angular_speed)
        quiet = torch.where(stationary, quiet + 1, 0)
        pose_still = (
            torch.linalg.vector_norm(pose.position - pose_reference.position, dim=-1) < 0.001
        ) & (pose.angular_distance(pose_reference).diagonal() < 0.01)
        pose_quiet = torch.where(pose_still, pose_quiet + 1, 0)
        pose_reference = Pose(
            torch.where(pose_still[:, None], pose_reference.position, pose.position),
            torch.where(pose_still[:, None], pose_reference.quaternion, pose.quaternion),
        )
        landed = (initial[:, 2] - pose.position[:, 2] >= 0.0005 if require_drop
                  else torch.ones_like(quiet, dtype=torch.bool))
        settled = quiet >= quiet_steps
        if not require_velocity_quiet:
            settled |= pose_quiet >= quiet_steps
        if step + 1 >= min_steps and (landed & settled).all():
            method = "velocity" if (quiet >= quiet_steps).all() else "pose"
            print(f"[SETTLE] {env.num_envs} objects settled by {method} in "
                  f"{(step + 1) * env.step_dt:.2f}s", flush=True)
            return
    raise RuntimeError(
        f"Objects did not settle within {max_seconds:g}s: "
        f"linear={torch.linalg.vector_norm(velocity[:, :3], dim=-1).max().item():.4f}m/s, "
        f"angular={torch.linalg.vector_norm(velocity[:, 3:], dim=-1).max().item():.4f}rad/s")


def execute_trajectory(env, app, plan, phase: str | None, action,
                       settle_seconds: float = 1.0, repeat_factor: int = 1,
                       max_tracking_error: float = 0.05, tracking_wait_s: float = 0.0,
                       interpolate_repeats: bool = False) -> None:
    """Replay the planner's timed joint trajectory through the configured action term."""
    import torch

    if not math.isfinite(tracking_wait_s) or tracking_wait_s < 0:
        raise ValueError("tracking_wait_s must be finite and nonnegative")
    label = phase or "pose"
    robot = env.scene["robot"]
    arm = env.action_manager.get_term("arm_action")
    arm_ids, arm_names = robot.find_joints(arm.cfg.joint_names, preserve_order=arm.cfg.preserve_order)
    prefix = f"{phase}_" if phase else ""
    trajectory = getattr(plan, f"{prefix}interpolated_trajectory")
    lengths = getattr(plan, f"{prefix}interpolated_last_tstep").reshape(-1)
    positions = trajectory.reorder(arm_names).position
    batch = action.shape[0]
    if (positions.ndim < 3 or math.prod(positions.shape[:-2]) != batch
            or positions.shape[-1] != len(arm_names) or lengths.numel() != batch
            or not ((lengths > 0) & (lengths <= positions.shape[-2])).all()):
        raise RuntimeError(f"Invalid {label} trajectory shape={tuple(positions.shape)}, lengths={lengths}")
    positions = positions.reshape(batch, -1, len(arm_names))
    last = int(lengths.max().item())
    # Hold each robot's own endpoint when its trajectory finishes earlier.
    indices = torch.arange(last, device=positions.device)[None, :].minimum(lengths[:, None] - 1).long()
    positions = positions.gather(1, indices[..., None].expand(-1, -1, len(arm_names)))
    if not torch.isfinite(positions).all():
        raise RuntimeError(f"Invalid {label} trajectory")
    if repeat_factor < 1:
        raise ValueError("repeat_factor must be a positive integer")
    if getattr(env, "compact_trajectory", False):
        # Drop a waypoint only when every environment repeats its previous target.
        keep = torch.cat((torch.ones(1, device=positions.device, dtype=torch.bool),
                          (positions[:, 1:] != positions[:, :-1]).any(dim=(0, 2))))
        positions = positions[:, keep]
        if repeat_factor == 1 and positions.shape[1] > 1 and torch.allclose(
                arm.position_actions(positions[:, 0]), action[:, :arm.action_dim],
                rtol=0, atol=1e-6):
            positions = positions[:, 1:]
    if repeat_factor > 1:
        if interpolate_repeats:
            previous = torch.cat((positions[:, :1], positions[:, :-1]), dim=1)
            fraction = torch.arange(1, repeat_factor+1, device=positions.device,
                                    dtype=positions.dtype)/repeat_factor
            positions = (previous[:, :, None]+(positions-previous)[:, :, None]
                         *fraction[None, None, :, None]).flatten(1, 2)
        else:
            positions = positions.repeat_interleave(repeat_factor, dim=1)
    for target in positions.unbind(dim=1):
        action[:, :arm.action_dim] = arm.position_actions(target)
        step_env(env, app, action)
    hold(env, app, action, settle_seconds)
    joint_error = robot.data.joint_pos[:, arm_ids] - positions[:, -1]
    error = joint_error.abs().max().item()
    if tracking_wait_s and math.isfinite(error) and error > max_tracking_error:
        print(f"[TRACKING] {label}: error={error:.4f} rad; "
              f"holding endpoint for at most {tracking_wait_s:g}s", flush=True)
        for _ in range(math.floor(tracking_wait_s / env.step_dt)):
            step_env(env, app, action)
            joint_error = robot.data.joint_pos[:, arm_ids] - positions[:, -1]
            error = joint_error.abs().max().item()
            if not math.isfinite(error) or error <= max_tracking_error:
                break
    if not 0.0 <= error <= max_tracking_error:
        index = int(joint_error.abs().amax(dim=-1).argmax().item())
        joint_errors = dict(zip(arm_names, joint_error[index].tolist()))
        raise RuntimeError(
            f"{label}: arm tracking error {error:.4f} rad; env={index}; joint errors={joint_errors}"
        )
    print(f"[EXECUTE] {label}: {positions.shape[1]} trajectory steps "
          f"({last} planned, repeat={repeat_factor}), settle={settle_seconds:g}s, "
          f"{batch} environments, error={error:.4f} rad", flush=True)


def verify_grasp(env, app, action, initial_z, *,
                 min_height: float = 0.15, max_distance: float = 0.035,
                 min_opening: float = 0.01, object_center_local=None, seconds: float = 1.0) -> None:
    """Require a lifted object close to the TCP throughout the requested hold."""
    if not math.isfinite(seconds) or seconds < env.step_dt:
        raise ValueError("seconds must be finite and cover at least one control step")
    import torch
    from curobo.types import Pose

    robot, cube = env.scene["robot"], env.scene["object"]
    center_local = (None if object_center_local is None else
                    object_center_local.expand_as(cube.data.root_pos_w))
    geometry = getattr(env.cfg, "grasp_geometry", None)
    if geometry is None:
        finger_ids, _ = robot.find_joints(env.cfg.actions.gripper_action.joint_names)
    for _ in range(round(seconds / env.step_dt)):
        step_env(env, app, action)
        tcp = env.scene["ee_frame"].data.target_pos_w[:, 0]
        center = cube.data.root_pos_w
        if center_local is not None:
            center = Pose(center, cube.data.root_quat_w).transform_points(center_local)
        heights = center[:, 2] - initial_z
        distances = torch.linalg.vector_norm(tcp - center, dim=-1)
        if geometry is not None:
            openings, touching = geometry(env)
            if not touching.all():
                raise RuntimeError("Grasp failed: both grasp contacts must touch the object")
        else:
            joint_openings = robot.data.joint_pos[:, finger_ids].sum(dim=-1)
            if (joint_openings <= 0.001).any():
                raise RuntimeError("Grasp failed: fingers reached the closed joint limit")
            openings = joint_openings + env.cfg.grasp_width_offset
        telemetry = torch.stack((heights, distances, openings), dim=-1).cpu().tolist()
        for env_id, values in enumerate(telemetry):
            try:
                check_grasp(*values, min_height=min_height,
                            max_distance=max_distance, min_opening=min_opening)
            except RuntimeError as exc:
                raise RuntimeError(f"Environment {env_id}: {exc}") from exc
    for env_id, (height, _, opening) in enumerate(telemetry):
        print(f"[SUCCESS] Physical grasp held for {seconds:g}s; env={env_id}, lift={height:.3f}m, width={opening:.3f}m", flush=True)
    print(f"[BATCH] {env.num_envs}/{env.num_envs} grasps verified", flush=True)

