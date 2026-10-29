"""Robot-only attempts for Extra 10--14; native success remains authoritative.

These experts use teacher diagnostics for collection. They neither write task
body state nor change materials, robot drives or the delivered gel model.
"""

from __future__ import annotations

import math
from numbers import Real

import numpy as np


def _state(env):
    timeout = env.task["episode_config"]["timeout_s"]
    if env.extra_step_counter*env.step_dt >= timeout:
        raise RuntimeError("Contact expert exceeded the task timeout")
    invalid = getattr(env.progress, "invalid_reason", None)
    if invalid:
        first = getattr(env.progress, "first_unsafe_contact", None)
        if first:
            violations = ", ".join(f"{key}={entry['value']:.9g} (limit {entry['limit']:.9g})"
                                   for key, entry in first["violations"].items())
            raise RuntimeError(f"Contact trajectory invalid: {invalid}; "
                               f"physics sample {first['sample_id']}: {violations}")
        raw = env.read_task_state()
        print("CONTACT INVALID", invalid,
              {key: round(float(value), 6) for key, value in raw.items()
               if isinstance(value, Real) and not isinstance(value, bool)}, flush=True)
        raise RuntimeError(f"Contact trajectory invalid: {invalid}")
    state = env.read_task_state()
    for key, value in state.items():
        if isinstance(value, (Real, list, tuple, np.ndarray)) and not np.isfinite(value).all():
            raise RuntimeError(f"Contact expert received nonfinite telemetry: {key}")
    return state


def _move(env, motion, position, yaw, *, duration, quaternion, converge=False, grasp=False,
          planned=False, payload=False, rotation_tolerance=.05, sliding=False, repeat_factor=1):
    def check(sample):
        _sliding_load(sample) if sliding else _grasp_load(env, sample)
    state = _state(env)
    if grasp:
        check(state)
    original_step = motion.step
    def checked_step():
        _state(env)
        result = original_step()
        sample = _state(env)
        if grasp:
            check(sample)
        return result
    motion.step = checked_step
    try:
        options = {"duration": duration, "quaternion": quaternion, "converge": converge,
                   "rotation_tolerance": rotation_tolerance}
        if planned:
            motion.move_planned(np.asarray(position), yaw, payload=payload, repeat_factor=repeat_factor, **options)
        else:
            # Keep joint centering from shifting an established loaded contact.
            if (env.task["episode_config"]["task_type"] == "wipe_whiteboard"
                    or env.task["episode_config"]["task_type"] == "push" and env.expert_phase >= 2
                    or env.task["episode_config"]["task_type"] in {"roll", "hang", "slip", "stand", "press_slide"} and env.expert_phase >= 1):
                options["center_joints"] = False
            motion.move(np.asarray(position), yaw,
                        interpolate=10 <= env.task["task_id"] <= 16, **options)
    finally:
        motion.step = original_step
    state = _state(env)
    if grasp:
        check(state)
    return state


def _step(env, motion):
    _state(env)
    motion.step()
    return _state(env)


def _hold(env, motion, seconds, *, grasp=False):
    for _ in range(math.ceil(seconds/env.step_dt)):
        state = _step(env, motion)
        if grasp:
            _grasp_load(env, state)
    return state


def _housing_load(state):
    housing = state.get("housing_total_force_n", state.get("housing_force_n", 0.))
    gel = state.get("gel_total_force_n", state.get("gel_force_n", 0.))
    if housing > .05 and housing/max(gel+housing, 1e-12) > .1:
        print(f"HOUSING gel={gel:.4f} housing={housing:.4f} "
              f"l/r={state.get('left_gel_force_n', 0.):.4f}/{state.get('right_gel_force_n', 0.):.4f} "
              f"cmd={state.get('gripper_command', float('nan')):.4f} "
              f"obj={np.round(state.get('object_position_w_m', [0, 0, 0]), 5).tolist()}", flush=True)
        raise RuntimeError("Contact expert cannot use the sensor housing as a grasp or pushing surface: "
                           f"housing={housing:.6g} N > 0.05 N, "
                           f"share={housing/max(gel+housing, 1e-12):.2%} > 10%, gel={gel:.6g} N")


def _carrying_preload(env):
    # Delivered gel (0.1/0.2) and task (0.6/0.5) friction combine by average:
    # both static and dynamic coefficients are 0.35. Reserve 20% over gravity.
    return 1.2*env.task["objects"]["mechanism"]["mass_kg"]*9.81/(2*.35)


def _grasp_load(env, state):
    threshold = env.task["success_config"]["contact_threshold_n"]
    if min(state["left_gel_force_n"], state["right_gel_force_n"]) < threshold:
        print(f"GRASP LOST l/r={state['left_gel_force_n']:.4f}/{state['right_gel_force_n']:.4f} "
              f"housing={state.get('housing_total_force_n', 0.):.4f} "
              f"obj={np.round(state['object_position_w_m'], 5).tolist()} "
              f"cmd={state.get('gripper_command', float('nan')):.4f}", flush=True)
        raise RuntimeError("Contact expert lost its two-gel physical grasp")
    _housing_load(state)


def _sliding_load(state):
    # A sliding contact can carry less than the initial grasp preload while
    # native target pairs and tangential reaction still prove continuous touch.
    # The hex knob can flicker one pad below the instantaneous presence test
    # between control samples; the native gel_contact_gap_s gate already fails
    # the episode if the physical two-pad gap exceeds 10 ms, so only a total
    # contact loss is rejected here.
    if not (state["left_gel_contact_present"] or state["right_gel_contact_present"]):
        raise RuntimeError("Contact expert lost both gel contacts during sliding")
    _housing_load(state)


def _balance_grasp(env, motion, *, limit=.001, step=.0001, relative=.1):
    """Center a pinch from measured pad loads, keeping jaw closure unchanged."""
    from scipy.spatial.transform import Rotation

    if not 0 < step <= limit:
        raise ValueError("Centering step must lie inside its bounded stroke")
    origin = axis = quaternion = None
    offset, step_size, previous_error = 0., step, 0.
    rounds = 2*math.ceil(limit/step)
    for index in range(rounds+1):
        state = _state(env)
        _housing_load(state)
        left, right = state["left_gel_force_n"], state["right_gel_force_n"]
        error = left-right
        if abs(error) <= max(.008, relative*(left+right)):
            return
        if index == rounds:
            break
        if origin is None:
            robot = env.scene["robot"]
            quaternion = robot.data.body_quat_w[0, robot.body_names.index("panda_hand")].cpu().tolist()
            frame = Rotation.from_quat([*quaternion[1:], quaternion[0]])
            axis = frame.apply([0., 1., 0.])
            yaw = frame.as_euler("xyz")[2]
            origin = env.scene["ee_frame"].data.target_pos_w[0, 0].cpu().numpy().copy()
        # A one-step sign flip means the correction passed the force balance;
        # shrink the step so it can settle inside the remaining force band.
        if previous_error*error < 0:
            step_size = max(step*.05, step_size*.5)
        previous_error = error
        # Translating toward the more heavily loaded left (+Y) pad unloads it
        # while loading the opposing pad. Limit the whole correction to the
        # caller's stroke; gears need several millimetres where the 2 mm slip
        # hand-translation gate is not in force.
        offset = float(np.clip(offset+math.copysign(step_size, error), -limit, limit))
        _move(env, motion, origin+axis*offset, yaw, duration=.15, quaternion=quaternion)
    raise RuntimeError("Contact expert could not balance the gel preload inside the bounded centering stroke")


def _pinch_rod(env, motion):
    """Reach a weight-derived preload through the actual two-pad reactions."""
    target = _carrying_preload(env)
    command, quiet = float(env.action[0, -1]), 0
    for _ in range(round(18./env.step_dt)):
        state = _state(env)
        _housing_load(state)
        if state["robot_force_n"] > state["gel_total_force_n"]+state["housing_total_force_n"]+.05:
            raise RuntimeError("Rod touched another robot body during preload")
        left, right = state["left_gel_force_n"], state["right_gel_force_n"]
        quiet = quiet+1 if min(left, right) >= target else 0
        if quiet >= 6:
            _hold(env, motion, .4, grasp=True)
            state = _state(env)
            if min(state["left_gel_force_n"], state["right_gel_force_n"]) >= target:
                return
            quiet = 0
        if max(left, right) > .12 and abs(left-right) > max(.03, .15*max(left, right)):
            _balance_grasp(env, motion, limit=.0005, step=.00005, relative=.08)
        elif min(left, right) < target:
            command = max(-1., command-(.025 if max(left, right) < .02 else .002))
            env.action[:, -1] = command
        _step(env, motion)
    raise RuntimeError("Rod did not reach its friction-derived carrying preload")


def _prepare_slip_lift(env, motion, yaw, quaternion):
    """Settle and align below the native 30 mm reference-arming clearance."""
    from scipy.spatial.transform import Rotation

    frame = env.scene["ee_frame"].data
    point = frame.target_pos_w[0, 0].cpu().numpy().copy()
    _move(env, motion, point+[0., 0., .008], yaw, duration=2., quaternion=quaternion,
          converge=True, grasp=True, rotation_tolerance=.015)
    if _state(env)["slip_armed"]:
        raise RuntimeError("Slip reference armed during the low departure")
    _hold(env, motion, 6., grasp=True)
    for _ in range(4):
        state = _state(env)
        if state["rod_axis_tilt_rad"] <= .005:
            break
        q = state["object_quaternion_wxyz"]
        axis = Rotation.from_quat([*q[1:], q[0]]).apply([0., 0., 1.])
        up = np.array([0., 0., 1. if axis[2] >= 0 else -1.])
        vector = np.cross(axis, up)
        angle = math.atan2(float(np.linalg.norm(vector)), float(np.dot(axis, up)))
        if angle > .1:
            raise RuntimeError("Departure rod axis exceeded the bounded wrist correction")
        frame = env.scene["ee_frame"].data
        q = frame.target_quat_w[0, 0].cpu().tolist()
        rotation = Rotation.from_rotvec(vector/np.linalg.norm(vector)*angle)*Rotation.from_quat([*q[1:], q[0]])
        quaternion = rotation.as_quat()[[3, 0, 1, 2]].tolist()
        point = frame.target_pos_w[0, 0].cpu().numpy().copy()
        _move(env, motion, point, yaw, duration=1., quaternion=quaternion,
              converge=True, grasp=True, rotation_tolerance=.003)
        _hold(env, motion, 6., grasp=True)
        if _state(env)["slip_armed"]:
            raise RuntimeError("Slip reference armed during the departure correction")
    if _state(env)["rod_axis_tilt_rad"] > .01:
        raise RuntimeError("Departure correction did not align the measured rod")
    return env.scene["ee_frame"].data.target_quat_w[0, 0].cpu().tolist()


def _pinch(env, motion, *, minimum_preload=0., fine_step=.004):
    """Approach a modest measured preload through existing position commands."""
    kind = env.task["episode_config"]["task_type"]
    if kind == "slip":
        return _pinch_rod(env, motion)
    # Bound gear preload to retain traction without closing the case clearance.
    target = max(minimum_preload, .095 if kind == "gears" else _carrying_preload(env))
    if kind == "wipe_whiteboard":
        target = max(minimum_preload, _carrying_preload(env)*(1.02/1.2))
    # Latch below the peak target before closure rotates a hex corner into a case.
    stop = target - (.018 if kind == "gears" else 0.)
    # The latched aperture settles below its peak load. Accept on the settled
    # floor instead of waiting for a peak the aperture never sustains.
    settled = target - (.025 if kind == "gears" else 0.)
    # Correct hub/tube tracking error; keep gear centering inside the case recess.
    limit, step = ((.0025, .00025) if kind == "gears" else (.004, .0002))
    if kind == "wipe_whiteboard":
        step = .00005  # The eraser's balanced contact window is below 0.2 mm.
    command = float(env.action[0, -1])
    quiet, best, level = 0, 0., 0.
    for _ in range(math.ceil((8. if kind == "gears" else 12.)/env.step_dt)):
        state = _state(env)
        _housing_load(state)
        force = min(state["left_gel_force_n"], state["right_gel_force_n"])
        best = max(best, force)
        # A coarse step lands with a short impact spike; latching the aperture
        # from one sample either closes into the pad drop or freezes short of
        # the load. Decide on a smoothed load instead.
        level = force if level == 0. else .75*level+.25*force
        if level >= settled:
            quiet += 1
        else:
            quiet = 0
            # One pad routinely reaches the hub or tube first. Centering before
            # the measured load difference grows keeps the opposite pad in gel
            # contact instead of transferring the grasp to the sensor case.
            if (max(state["left_gel_force_n"], state["right_gel_force_n"]) >= .02
                    and abs(state["left_gel_force_n"]-state["right_gel_force_n"])
                    > .2*max(state["left_gel_force_n"], state["right_gel_force_n"])):
                _balance_grasp(env, motion, limit=limit, step=step, relative=.05)
                state = _state(env)
                _housing_load(state)
                force = min(state["left_gel_force_n"], state["right_gel_force_n"])
                best = max(best, force)
                level = .75*level+.25*force
        if force < target and level < stop:
            # The knob is a free passive joint: creeping the last millimetres
            # lets the applied torque rotate a hex corner under one pad until
            # that pad drops out. Use fine steps as soon as either pad carries
            # measurable load; the coarse 0.025 step jumps the one-step band.
            fine = max(state["left_gel_force_n"], state["right_gel_force_n"]) >= .5*target
            command = max(-1., command-(fine_step if fine else .025))
            env.action[:, -1] = command
        elif quiet >= (6 if kind == "gears" else 2):
            # A final centering slide can spin the gear past the native active
            # gate and interrupt pad contact; only recenter when the measured
            # split still matters.
            left, right = state["left_gel_force_n"], state["right_gel_force_n"]
            if kind != "gears" or abs(left-right) > .2*max(left, right):
                _balance_grasp(env, motion, limit=limit, step=step, relative=.05)
            _grasp_load(env, _state(env))
            if kind == "gears":
                _hold(env, motion, .25, grasp=True)
            return
        _step(env, motion)
    raise RuntimeError(f"Contact expert could not establish a two-gel preload of "
                       f"{target:.3f} N per pad (best measured {best:.3f} N)")


def _lift_check(env, before, state):
    _grasp_load(env, state)
    if state["object_position_w_m"][2]-before[2] < .03:
        raise RuntimeError("Contact expert failed to lift the object through its gel grasp")


def _settle_success(env, motion, *, grasp=False, timeout=1.5, confirm_steps=1):
    env.expert_phase = 4
    quiet = 0
    for _ in range(math.ceil(timeout/env.step_dt)):
        state = _step(env, motion)
        if grasp:
            _grasp_load(env, state)
        quiet = quiet+1 if env.progress.result.get("success", False) else 0
        if quiet >= confirm_steps:
            return
    raise RuntimeError(f"Contact expert did not complete the physical success contract: {env.progress.result}")


def _surface_pose(path, quaternions, fraction):
    """Interpolate a surface-path pose using measured object progress."""
    from scipy.spatial.transform import Rotation, Slerp

    coordinate = 1+np.clip(fraction, 0., 1.)*(len(path)-2)
    lower = min(int(math.floor(coordinate)), len(path)-1)
    upper = min(lower+1, len(path)-1)
    amount = coordinate-lower
    position = (1.-amount)*np.asarray(path[lower][0])+amount*np.asarray(path[upper][0])
    yaw_delta = (path[upper][1]-path[lower][1]+math.pi)%(2.*math.pi)-math.pi
    yaw = path[lower][1]+amount*yaw_delta
    rotations = Rotation.from_quat(np.asarray([quaternions[lower], quaternions[upper]])[:, [1, 2, 3, 0]])
    quaternion = Slerp([0., 1.], rotations)([amount]).as_quat()[0][[3, 0, 1, 2]]
    return position, yaw, quaternion.tolist()


def _roll_two_pad(env, motion, path, quaternions, layout):
    """Translate a measured pinch while the can rolls on the unchanged table."""
    inward = layout.apply([1., 0., 0.])
    quaternion, yaw = quaternions[1], path[1][1]
    center = np.asarray(_state(env)["object_position_w_m"])
    env.action[:, -1] = 1.
    env.expert_phase = 0
    _move(env, motion, center+[0., 0., .08], yaw, duration=2.,
          quaternion=quaternion, converge=True, planned=True)
    _move(env, motion, center, yaw, duration=2., quaternion=quaternion, converge=True)
    env.expert_phase = 1
    # Center only across the open jaws, preserving the approach height.
    from scipy.spatial.transform import Rotation
    axis = Rotation.from_quat([*quaternion[1:], quaternion[0]]).apply([0., 1., 0.])
    for _ in range(math.ceil(1./env.step_dt)):
        body = np.asarray(_state(env)["object_position_w_m"])
        point = env.scene["ee_frame"].data.target_pos_w[0, 0].cpu().numpy().copy()
        error = float(np.dot(body-point, axis))
        if abs(error) <= .0002:
            break
        _move(env, motion, point+axis*error, yaw, duration=env.step_dt, quaternion=quaternion,
              rotation_tolerance=None)
    else:
        raise RuntimeError("Rolling pinch did not center before closing")
    for _ in range(math.ceil(12./env.step_dt)):
        state = _state(env)
        _housing_load(state)
        left, right = state["left_gel_force_n"], state["right_gel_force_n"]
        if min(left, right) >= .01 and left+right >= .08:
            break
        advance = False
        if left+right >= .025 and abs(left-right) > .015:
            _balance_grasp(env, motion, limit=.001, step=.00005, relative=.15)
        else:
            step = (.04 if min(left, right) >= .003 and left+right < .08 else
                    .01 if max(left, right) < .02 else .001)
            env.action[:, -1] = max(-1., float(env.action[0, -1])-step)
            advance = step == .04
        if advance:
            point = env.scene["ee_frame"].data.target_pos_w[0, 0].cpu().numpy().copy()
            _move(env, motion, point+inward*.002, yaw, duration=env.step_dt,
                  quaternion=quaternion, rotation_tolerance=None)
        else:
            _step(env, motion)
    else:
        raise RuntimeError("Rolling pinch did not reach a balanced supported preload")
    print("ROLLING PINCH", _state(env), flush=True)
    state = _state(env)
    point = env.scene["ee_frame"].data.target_pos_w[0, 0].cpu().numpy().copy()
    body = np.asarray(state["object_position_w_m"])
    offset = point-body
    start = body.copy()
    goal = env.task["goals"][0]
    env.expert_phase = 2
    for index in range(math.ceil(45./env.step_dt)):
        state = _state(env)
        _housing_load(state)
        if state["roll_angle_rad"] >= goal["target_rad"]-.03:
            break
        total = state["left_gel_force_n"]+state["right_gel_force_n"]
        if total < .08:
            env.action[:, -1] = max(-1., float(env.action[0, -1])-.001)
        elif total > .20:
            env.action[:, -1] = min(1., float(env.action[0, -1])+.001)
        body = np.asarray(state["object_position_w_m"])
        travel = float(np.dot(body-start, inward))
        requested = min((index+1)*.030*env.step_dt, travel+.0015)
        target = start+inward*requested+offset
        _move(env, motion, target, yaw, duration=env.step_dt, quaternion=quaternion,
              rotation_tolerance=None)
    else:
        raise RuntimeError("Rolling pinch did not complete the half-turn")
    _settle_success(env, motion, timeout=5., confirm_steps=3)


def _surface(env, motion, path, quaternions, layout):
    """Push along the shared curve using measured surface progress."""
    goal = env.task["goals"][0]
    def normal(yaw):
        angle = yaw-math.pi/2
        value = np.array([math.cos(angle), math.sin(angle), 0.])
        if np.dot(value, layout.apply([1., 0., 0.])) < 0:
            value = -value
        return value
    inward = normal(path[1][1])
    env.action[:, -1] = 1.
    env.expert_phase = 0
    setback = .006
    hover = path[0][0]-setback*inward
    _move(env, motion, hover, path[0][1], duration=2., quaternion=quaternions[0], converge=True, planned=True)
    approach = path[1][0]-setback*inward
    # Finish the approach clear of the surface so preload controls first touch.
    _move(env, motion, approach, path[1][1], duration=2., quaternion=quaternions[1], converge=True)
    env.expert_phase = 1
    # Keep preload and tracking offsets relative to the same surface point.
    offset = -setback
    motion.normal_correction_m = 0.
    touch_origin = np.array(_state(env)["object_position_w_m"])
    # Sliding drag is below breakaway load: hand off to the curved path once
    # contact is sustained, even if the initial force command is not reached.
    touch_target = env.task["success_config"]["contact_threshold_n"]
    follow = .004
    duration = env.step_dt
    # Establish contact with the same control-rate force feedback used for
    # tracking; a coarse first touch can spike over the contact gate and then
    # unload before the next command while the free object starts moving.
    motion.surface_normal = inward
    motion.contact_force_target_n = .08
    motion.contact_force_limit_n = .09
    motion.normal_correction_limit_m = .005
    confirm = 0
    for _ in range(int(math.ceil(12./duration))):
        state = _state(env)
        _housing_load(state)
        # Require two control samples while native history checks every substep.
        confirm = confirm+1 if state["gel_force_n"] >= touch_target else 0
        if confirm >= 2:
            break
        # Establish load before breakaway motion unloads the first touch.
        offset += .00025
        drift = np.asarray(state["object_position_w_m"])-touch_origin
        drift -= inward*np.dot(drift,inward)
        _move(env, motion, path[1][0]+offset*inward+np.clip(drift,-follow,follow),
              path[1][1], duration=duration, quaternion=quaternions[1])
    else:
        raise RuntimeError("Contact expert could not touch the object with a gel surface")
    origin = np.array(_state(env)["object_position_w_m"])
    if getattr(env, "contact_runtime", None) is not None:
        origin = np.asarray(env.contact_runtime.position)
    # Keep enough force available to restart after static friction arrests
    # the block, while bounding compression below the housing contact load.
    target_force = .08
    upper_force = .09
    motion.contact_force_target_n = target_force
    motion.contact_force_limit_n = upper_force
    env.expert_phase = 2
    stalled, index = 0, 2
    tracking_dt = env.step_dt
    previous_position = np.asarray(_state(env)["object_position_w_m"]).copy()
    try:
        while index < len(path):
            state = _state(env)
            _housing_load(state)
            progress = state["path_u"]
            # Leave half the completion margin for the loaded target to settle.
            # Replacing it with measured joints abruptly unloads the gel.
            if progress >= .5*(1.+goal["coverage"]):
                break
            # A free object can cross several nominal waypoints in one move.
            # Catch up to its measured position instead of retracting the pad
            # toward a stale waypoint and losing the loaded contact.
            index = max(index, min(len(path)-1, math.floor(progress*(len(path)-2))+2))
            waypoint_fraction = (index-1)/(len(path)-2)
            lead = .5/(len(path)-2)
            # Keep a near-threshold contact from unloading during the next
            # wrist command. The native continuity gate remains active.
            if (progress >= .5 and state["gel_force_n"] <
                    1.05*env.task["success_config"]["contact_threshold_n"]):
                motion.normal_correction_limit_m = .006
                motion.normal_correction_m = min(.006, motion.normal_correction_m+.0005)
            arc_per_fraction = math.hypot(goal["length_m"],
                2.*math.pi*goal["amplitude_m"]*math.cos(2.*math.pi*progress))
            lead = min(lead, .005*tracking_dt/arc_per_fraction)
            fraction = min(waypoint_fraction, progress+lead)
            surface_point, surface_yaw, surface_quaternion = _surface_pose(path, quaternions, fraction)
            expected = np.array([goal["length_m"]*fraction,
                                 goal["amplitude_m"]*math.sin(2*math.pi*fraction), 0.])
            tangent = layout.apply([goal["length_m"],
                                    2.*math.pi*goal["amplitude_m"]*math.cos(2.*math.pi*fraction), 0.])
            tangent /= np.linalg.norm(tangent)
            inward = normal(surface_yaw)
            motion.surface_normal = inward
            error = origin+layout.apply(expected)-np.asarray(state["object_position_w_m"])
            error[2] = 0.
            error = np.clip(error, -follow, follow)
            # Follow measured progress and correct only across the surface
            # path. Adding along-path lag makes the pad drive ahead when the
            # object slows and breaks gel contact or rolls into sliding.
            cross_track = np.cross([0., 0., 1.], tangent)
            cross_track /= np.linalg.norm(cross_track)
            error = cross_track*float(np.dot(error, cross_track))
            from scipy.spatial.transform import Rotation

            # Follow the displaced round surface and steer its normal
            # toward the curve. Moving the pad toward the curve without
            # turning it puts contact on the opposite side of the circle
            # and pushes the block farther away.
            # At three quarters of the cross-track limit, turn through
            # the authored friction cone (average mu=0.35). Transverse
            # pad friction cannot reverse the force correcting the error.
            steering_length = .75*goal["cross_track_m"]/.35
            # Compensate the measured object's travel during wrist tracking.
            position = np.asarray(state["object_position_w_m"])
            speed = max(0., float(np.dot(position-previous_position, tangent))/tracking_dt)
            ahead = min(1., fraction+.2*speed/arc_per_fraction)
            future = layout.apply([goal["length_m"],
                2.*math.pi*goal["amplitude_m"]*math.cos(2.*math.pi*ahead), 0.])
            future /= np.linalg.norm(future)
            heading = future+error/steering_length
            heading /= np.linalg.norm(heading)
            angle = math.atan2(np.cross(inward, heading)[2], np.dot(inward, heading))
            turn = Rotation.from_euler("z", angle)
            center = origin+layout.apply(expected)
            surface_point = center+turn.apply(surface_point-center)-error
            surface_yaw += angle
            q = surface_quaternion
            surface_quaternion = (turn*Rotation.from_quat([*q[1:], q[0]])).as_quat()[[3, 0, 1, 2]].tolist()
            inward = heading
            motion.surface_normal = inward
            cap = .006
            offset_step = .00025*tracking_dt
            offset = max(-setback, min(cap, offset+(offset_step if state["gel_force_n"] < target_force else
                                                  -offset_step if state["gel_force_n"] > upper_force else 0.)))
            target = surface_point+inward*offset
            position = np.asarray(state["object_position_w_m"])
            # Follow the preceding control interval's forward travel so
            # the freely sliding block cannot outrun its loaded pad
            # between the 20 Hz commands. Keep force and stroke limits.
            target += inward*np.clip(np.dot(position-previous_position, inward), 0., .001)
            previous_position = position.copy()
            # Refresh steering at control rate, with only 5 mm/s of nominal
            # advance. Waiting for each moving target to converge leaves the
            # previous contact normal loaded while the free block drifts.
            _move(env, motion, target, surface_yaw, duration=tracking_dt,
                  quaternion=surface_quaternion, rotation_tolerance=None)
            measured = _state(env)
            next_progress = measured["path_u"]
            stalled = stalled+tracking_dt if next_progress <= progress+1e-4 else 0.
            if stalled+1e-12 >= 4.:
                raise RuntimeError("Contact expert produced no measured object progress despite robot motion")
            if next_progress > 1.03:
                raise RuntimeError("Contact expert overshot the physical path")
            # Keep the pad moving along its nominal surface path while the free
            # body trails by at most 1.5 waypoints; a stationary pad cannot pull
            # the object forward, and the stall gate above still rejects a lost
            # contact. The final target still waits for measured completion.
            if (next_progress >= waypoint_fraction-1.5/(len(path)-2)
                    and (index < len(path)-1 or next_progress >= .999)):
                index += 1
    finally:
        motion.surface_normal = None
        motion.normal_correction_limit_m = .001
        motion.contact_force_limit_n = None
    _settle_success(env, motion)


def _gears(env, motion, path, quaternions):
    env.action[:, -1] = 1.
    env.expert_phase = 0
    for index in (0, 1):
        _move(env, motion, *path[index], duration=2., quaternion=quaternions[index], converge=True, planned=index == 0)
    env.expert_phase = 1
    _pinch(env, motion)
    # Orbiting an off-centre grip swings the pads around the hub at a radius;
    # the hex then walks out from under one pad and the case follows it in.
    # Centre the TCP on the measured knob axis before the yaw sweep.
    from benchmark.tasks.extra.interaction import grasp_point
    tcp = env.scene["ee_frame"].data
    point = tcp.target_pos_w[0,0].cpu().numpy().copy()
    center = grasp_point(env)
    if np.linalg.norm((center-point)[:2]) > .0003:
        quaternion = tcp.target_quat_w[0,0].cpu().tolist()
        _move(env, motion, center, 0., duration=.8, quaternion=quaternion, grasp=True)
        point = env.scene["ee_frame"].data.target_pos_w[0,0].cpu().numpy().copy()
    env.expert_phase = 2
    goal = env.task["goals"][0]
    # Limit wrist speed and lead so the pads do not cross the hex corners
    # into the housing during the turn.
    duration = goal["input_target_rad"]/.12
    speed = goal["input_target_rad"]/duration
    _ramp_gear_turn(env, motion, point, goal["input_target_rad"], duration)
    for _ in range(3):
        error = goal["input_target_rad"]-_state(env)["input_angle_rad"]
        if abs(error) <= goal["input_tolerance_rad"]:
            break
        correction = max(-math.radians(10), min(math.radians(10), error))
        point = env.scene["ee_frame"].data.target_pos_w[0,0].cpu().numpy().copy()
        _ramp_gear_turn(env, motion, point, correction, abs(correction)/speed)
    _settle_success(env, motion, grasp=True)


def _ramp_gear_turn(env, motion, point, angle, duration):
    """Bound hand lead using the measured input wheel; never drive gear state."""
    from scipy.spatial.transform import Rotation
    from benchmark.tasks.extra.interaction import grasp_point

    robot = env.scene["robot"]
    q = robot.data.body_quat_w[0, robot.body_names.index("panda_hand")].cpu().tolist()
    start = Rotation.from_quat([*q[1:], q[0]])
    if not math.isfinite(angle) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("Gear turn needs a finite angle and positive duration")
    initial = previous = _state(env)["input_angle_rad"]
    point = np.asarray(point).copy()
    center = grasp_point(env)
    offset = point-center
    axial_offset = 0.
    direction = math.copysign(1., angle)
    speed, command, stalled = abs(angle)/duration, 0., 0
    # The knob hex grows from its 30 mm flats toward 34.6 mm corners as the
    # hand leads. A fixed aperture unloads one pad partway through the turn,
    # halving the drive torque; follow the measured two-pad load instead.
    aperture = float(env.action[0, -1])
    aperture_floor = max(-1., aperture-.010)
    # More wrist lead walks the pads over the hex corners into the case.
    # Keep the checked robot-target window and fail if its torque is inadequate.
    max_lead = .16
    steps = max(1, math.ceil((2*duration+15.)/env.step_dt))
    window = max(1, round(1.8/env.step_dt))
    for index in range(1, steps+1):
        sample = _state(env)
        if env.progress.result.get("success", False):
            return
        low = min(sample["left_gel_force_n"], sample["right_gel_force_n"])
        if low < .06:
            aperture = max(aperture_floor, aperture-.003)
        elif low > .11:
            aperture = min(1., aperture+.001)
        env.action[:, -1] = aperture
        measured = direction*(sample["input_angle_rad"]-initial)
        if measured >= abs(angle)-.001:
            return
        command = min(command+speed*env.step_dt, measured+max_lead, abs(angle)+max_lead)
        turn = direction*command
        orbit = Rotation.from_euler("z", turn)
        rotation = orbit*start
        x, y, z, w = rotation.as_quat()
        yaw = rotation.as_euler("xyz")[2]
        # Pad shear is the force on the sensor: positive Z means the hand is
        # pushing the hub down. Relax this parasitic load before it consumes
        # the available tangential traction. Change robot targets only.
        axial = sum(sample[f"{side}_gel_tangent_force_w_n"][2] for side in ("left", "right"))
        axial_offset = float(np.clip(axial_offset+np.clip(axial*.001,-.00005,.00005),-.001,.001))
        state = _move(env, motion, center+orbit.apply(offset)+[0.,0.,axial_offset], yaw, duration=env.step_dt,
                      quaternion=[w, x, y, z], grasp=True, sliding=True)
        if index % window == 0:
            current = state["input_angle_rad"]
            stalled = stalled+1 if direction*(current-previous) <= 1e-4 else 0
            if stalled >= 8:
                raise RuntimeError("Contact expert could not rotate the gear through its hub grasp")
            previous = current
    raise RuntimeError("Contact expert could not complete the measured gear turn through its hub grasp")


def _correct_cup_target(env, position, yaw):
    """Compensate bounded measured cup/hand drift before hook insertion."""
    from scipy.spatial.transform import Rotation
    from benchmark.tasks.extra.interaction import _rotation
    from benchmark.tasks.extra.contact_assets import asset_specs

    geometry = asset_specs()["extra_hanging_mug"]
    sample = _state(env)
    handle = np.asarray(sample["object_position_w_m"])+_rotation(
        sample["object_quaternion_wxyz"]).apply(geometry["handle_center_local_m"])
    tcp = env.scene["ee_frame"].data.target_pos_w[0, 0].cpu().numpy()
    offset = np.asarray(geometry["handle_center_local_m"])-geometry["handle_grip_center_local_m"]
    correction = tcp-handle+Rotation.from_euler("z", yaw).apply(offset)
    if np.linalg.norm(correction) > .012:
        raise RuntimeError("Held cup drift exceeded the bounded insertion correction")
    return np.asarray(position)+correction


def _hang(env, motion, path, quaternions):
    if len(path) != 7:
        raise ValueError("Hanging path must contain approach, grasp, lift, prethread, thread, lower and retreat")
    # The raised hollow grip and integral rails carry the cup through both gels.
    env.action[:, -1] = 1.
    env.expert_phase = 0
    for index in (0, 1):
        _move(env, motion, *path[index], duration=2., quaternion=quaternions[index], converge=True, planned=index == 0)
    env.expert_phase = 1
    _pinch(env, motion)
    motion.cup_carry_servo = True
    motion.cup_closure_floor = max(-1., float(env.action[0, -1])-.03)
    before = np.asarray(_state(env)["object_position_w_m"])
    state = _move(env, motion, *path[2], duration=3., quaternion=quaternions[2], converge=True, grasp=True,
                  planned=True, payload=True)
    _lift_check(env, before, state)
    env.expert_phase = 2
    for index in (3, 4, 5):
        if index in (4, 5):
            path[index] = (_correct_cup_target(env, *path[index]), path[index][1])
        if index == 5:
            # The hook load spiked above its 1.2-weight limit when the cup
            # reached the seat at 4 cm/s. Lower the final 3 mm at 1 mm/s so
            # native hook contact is established without an impact transient.
            start = env.scene["ee_frame"].data.target_pos_w[0, 0].cpu().numpy().copy()
            delta = np.asarray(path[index][0])-start
            distance=float(np.linalg.norm(delta))
            progress=0.
            for _ in range(max(1,math.ceil(distance/.00005))):
                if progress>=distance:
                    break
                state=_state(env)
                advance=.0005 if distance-progress>.004 and state["hook_up_force_n"]<1e-5 else .00005
                progress=min(distance,progress+advance)
                target=start+delta*(progress/max(distance,1e-12))
                state = _move(env, motion, target, path[index][1], duration=env.step_dt,
                              quaternion=quaternions[index], grasp=True)
        else:
            state = _move(env, motion, *path[index], duration=2., quaternion=quaternions[index], grasp=True,
                          converge=True, planned=index == 3, payload=index == 3)
        _grasp_load(env, state)
    goal = env.task["goals"][0]
    if not state["hook_through_handle"] or state["lowering_m"] < goal["minimum_lowering_m"]:
        raise RuntimeError("Contact expert did not physically thread and lower the mug before release")
    _unload_hook(env, motion, path[5][1])
    env.expert_phase = 3
    _release_hook(env, motion)
    # Let the released swing decay before retreat crosses the clearance envelope.
    for _ in range(math.ceil(5./env.step_dt)):
        _step(env,motion)
        if env.progress.result.get("success",False):
            break
    else:
        raise RuntimeError("Released mug did not earn its original stable hold")
    _move(env, motion, *path[6], duration=1., quaternion=quaternions[6], converge=True)
    _settle_success(env, motion)



def _unload_hook(env, motion, yaw):
    from scipy.spatial.transform import Rotation

    goal = env.task["goals"][0]
    weight = env.task["objects"]["mechanism"]["mass_kg"] * 9.81
    threshold = env.task["success_config"]["contact_threshold_n"]
    target = max(1.25 * threshold, 0.8 * weight / (2 * 0.35))
    if 2 * 0.35 * target >= weight:
        raise RuntimeError("This cup does not have a two-pad gravity-transfer load window")
    motion.cup_carry_servo = False
    _freeze_arm(env)
    _hold(env, motion, 0.3, grasp=True)
    frame = env.scene["ee_frame"].data
    anchor = frame.target_pos_w[0, 0].cpu().numpy().copy()
    q = frame.target_quat_w[0, 0].cpu().tolist()
    tangent = Rotation.from_quat([*q[1:], q[0]]).apply([1.0, 0.0, 0.0])
    tangent[2] = 0.0
    tangent /= np.linalg.norm(tangent)
    command = float(env.action[0, -1])
    quiet = 0
    settling = False
    for index in range(400):
        state = _state(env)
        _grasp_load(env, state)
        fraction = state["hook_up_force_n"] / weight
        if fraction > goal["max_hook_weight_fraction"]:
            raise RuntimeError("Gravity handoff overloaded the hook before release")
        loaded = fraction >= goal["min_hook_weight_fraction"] and state["other_support_force_n"] <= goal["max_other_support_force_n"]
        quiet = quiet + 1 if loaded else 0
        if quiet >= 3:
            _freeze_arm(env)
            _hold(env, motion, 0.5, grasp=True)
            state = _state(env)
            fraction = state["hook_up_force_n"] / weight
            if not goal["min_hook_weight_fraction"] <= fraction <= goal["max_hook_weight_fraction"]:
                raise RuntimeError("Hook transfer did not retain its full-weight support")
            return
        low = min(state["left_gel_force_n"], state["right_gel_force_n"])
        if low > target + 0.002:
            command = min(1.0, command + 0.0005)
        elif low < target - 0.004:
            command = max(-1.0, command - 0.00025)
        env.action[:, -1] = command
        if loaded:
            if not settling:
                _freeze_arm(env)
                settling = True
            _step(env, motion)
        else:
            settling = False
            offset = 0.00005 * math.sin(2 * math.pi * 4 * (index + 1) * env.step_dt) if fraction >= 0.2 else 0.0
            _move(env, motion, anchor + tangent * offset, yaw, duration=env.step_dt, quaternion=q, grasp=True)
    raise RuntimeError("Gravity handoff did not complete its unchanged hook load contract")


def _balance_hook_com(env, motion):
    """Balance the suspended cup before removing its remaining gel restraint."""
    hook, rotation = env.contact_runtime.pose("hook")
    axis = rotation.apply([1., 0., 0.])
    frame = env.scene["ee_frame"].data
    point = frame.target_pos_w[0, 0].cpu().numpy().copy()
    quaternion = frame.target_quat_w[0, 0].cpu().tolist()
    origin = point.copy()
    weight = env.task["objects"]["mechanism"]["mass_kg"]*9.81
    quiet = 0
    for _ in range(60):
        state = _state(env)
        _housing_load(state)
        if not state["hook_through_handle"]:
            raise RuntimeError("Cup lost its loaded hook during COM alignment")
        if state["hook_up_force_n"] < .8*weight:
            # A lateral correction can temporarily share weight with the
            # jaws. Hold the command until the loaded pivot is restored.
            quiet = 0
            _step(env, motion)
            continue
        center = env.scene["mechanism"].data.root_com_pos_w[0].cpu().numpy()
        error = float(np.dot(hook-center, axis))
        quiet = quiet+1 if abs(error) < .00005 else 0
        if quiet >= 3:
            return
        # The grip is above the shaft and COM below it, so their lateral
        # displacements have opposite signs. Use the fixed shaft centerline;
        # its surface contact centroid moves as the bail rotates.
        point -= axis*np.clip(error, -.000025, .000025)
        if np.linalg.norm(point-origin) > .0015+1e-9:
            raise RuntimeError("Cup COM correction exceeded its bounded 1.5 mm TCP stroke")
        _move(env, motion, point, 0., duration=env.step_dt, quaternion=quaternion, grasp=True)
    raise RuntimeError("Cup COM did not align beneath the shaft inside the bounded stroke")


def _release_hook(env, motion):
    goal = env.task["goals"][0]
    weight = env.task["objects"]["mechanism"]["mass_kg"] * 9.81
    success = env.task["success_config"]
    def supported(state):
        return (state["hook_through_handle"]
                and goal["min_hook_weight_fraction"] <= state["hook_up_force_n"]/weight <= goal["max_hook_weight_fraction"]
                and state["other_support_force_n"] <= goal["max_other_support_force_n"])
    def stationary(state):
        return (state["object_speed_m_s"] <= success["max_object_speed_m_s"]
                and state["angular_speed_rad_s"] <= success["max_angular_speed_rad_s"])
    _balance_hook_com(env, motion)
    _freeze_arm(env)
    command = float(env.action[0, -1])
    quiet = 0
    for _ in range(240):
        state = _state(env)
        _housing_load(state)
        if not state["hook_through_handle"]:
            raise RuntimeError("Mug lost its loaded hook before the jaws cleared")
        settled, loaded = stationary(state), supported(state)
        quiet = quiet + 1 if state["robot_force_n"] < 0.005 and loaded and settled else 0
        if quiet >= 3:
            # The hook has carried the stationary cup with no jaw load for
            # three samples. Clear the unloaded jaws without repeating the
            # fine opening increments used for the gravity handoff.
            while command < 1.0:
                command = min(1.0, command + 0.04)
                env.action[:, -1] = command
                state = _step(env, motion)
                _housing_load(state)
                if not supported(state):
                    raise RuntimeError("Mug lost its loaded hook support while opening the gripper")

            # Retrace the insertion direction only after the fingers are open.
            from scipy.spatial.transform import Rotation

            frame = env.scene["ee_frame"].data
            anchor = frame.target_pos_w[0, 0].cpu().numpy().copy()
            q = frame.target_quat_w[0, 0].cpu().tolist()
            yaw = Rotation.from_quat([*q[1:], q[0]]).as_euler("xyz")[2]
            _, rotation = env.contact_runtime.pose("hook")
            direction = rotation.apply([0., 1., 0.])
            withdrawal_step = .0002
            maximum_steps = math.ceil(.05/withdrawal_step)
            if state["release_clearance_m"] < goal["minimum_release_clearance_m"]:
                for step in range(1,maximum_steps+1):
                    target = anchor-direction*(withdrawal_step*step)
                    state = _move(env, motion, target, yaw,
                                  duration=env.step_dt, quaternion=q)
                    _housing_load(state)
                    if not supported(state):
                        raise RuntimeError("Mug lost its loaded hook while withdrawing along the insertion path")
                    if not stationary(state):
                        settled = 0
                        for _ in range(math.ceil(2./env.step_dt)):
                            state = _step(env, motion)
                            _housing_load(state)
                            if not supported(state):
                                raise RuntimeError("Mug lost its loaded hook while settling during withdrawal")
                            settled = settled+1 if stationary(state) else 0
                            if settled >= 3:
                                break
                        if settled < 3:
                            raise RuntimeError("Mug did not settle while withdrawing from the hook")
                    if state["release_clearance_m"] >= goal["minimum_release_clearance_m"]:
                        break
                else:
                    raise RuntimeError("Gripper did not clear the hook's jaw-release envelope")
            if state["robot_force_n"] >= goal["release_force_n"]:
                raise RuntimeError("Gripper remained loaded after withdrawing from the hook")
            return
        # Unload only in the existing stable-hang window; pause the jaws if the
        # cup starts moving or the hook gives up any of the required support.
        if loaded and settled:
            command = min(1.0, command + 0.0005)
            env.action[:, -1] = command
        _step(env, motion)
    raise RuntimeError("Loaded, settled jaw release did not clear native robot contact")



def _freeze_arm(env):
    from benchmark.backends.gsmini.environment import _hold_current_arm_action
    _hold_current_arm_action(env, env.action)


def _slip(env, motion, path, quaternions):
    env.action[:, -1] = 1.
    env.expert_phase = 0
    for index in (0, 1):
        # The 2 degree rod-tilt gate leaves less room than the default 0.05 rad
        # arrival tolerance; align the pads before they can torque the rod.
        _move(env, motion, *path[index], duration=2., quaternion=quaternions[index], converge=True,
              planned=index == 0, rotation_tolerance=.015)
    env.expert_phase = 1
    _pinch(env, motion)
    preload_command = float(env.action[0, -1])
    before = np.asarray(_state(env)["object_position_w_m"])
    # The low departure remains below reference arming. The remaining lift
    # follows one collision-checked, smoothly subdivided joint trajectory.
    lift_quaternion = _prepare_slip_lift(env, motion, path[2][1], quaternions[2])
    state = _move(env, motion, *path[2], duration=6., quaternion=lift_quaternion, grasp=True,
                  rotation_tolerance=.015, planned=True, payload=True, converge=True, repeat_factor=8)
    _lift_check(env, before, state)
    if not state["slip_armed"]:
        _balance_grasp(env, motion)
    _freeze_arm(env)
    # The freshly lifted rod still swings after the table contact breaks; the
    # 2 degree tilt gate applies while armed, so let the pendulum settle before
    # opening the fingers or the residual swing plus slip torque crosses it.
    settled = 0.
    minimum_tilt = math.inf
    for _ in range(math.ceil(6./env.step_dt)):
        state = _step(env, motion)
        _grasp_load(env, state)
        tilt = state["rod_axis_tilt_rad"]
        minimum_tilt = min(minimum_tilt, tilt)
        settled = settled+env.step_dt if state["slip_armed"] and tilt <= .006 else 0.
        if settled >= .4:
            break
    if settled < .4:
        raise RuntimeError(f"Contact expert could not settle the suspended rod before slip "
                           f"(minimum tilt {minimum_tilt:.4f} rad)")
    env.expert_phase = 2
    goal = env.task["goals"][0]
    target, tolerance = (goal["target_slip_m"], goal["tolerance_m"])
    open_limit = min(1.0, preload_command + 0.075)
    hold_command = max(-0.8, preload_command - 0.04)
    command = preload_command
    mode = "unload"
    quiet = 0
    held = 0
    for index in range(math.ceil(60.0 / env.step_dt)):
        state = _state(env)
        _sliding_load(state)
        if not state["slip_armed"] or not state["suspended"]:
            raise RuntimeError("Pulsed slip lost its unsupported native reference")
        if env.progress.result.get("success", False):
            env.expert_phase = 4
            return
        displacement, speed = (state["axial_slip_m"], max(0.0, state["axial_speed_m_s"]))
        if displacement > target + tolerance - 0.0005:
            raise RuntimeError("Pulsed slip exceeded its bounded arrest margin")
        stopping = speed * (2.0 * env.step_dt) + speed * speed / (2.0 * 0.5 * 9.81)
        if mode in ("unload", "slide"):
            if displacement + stopping >= target - 0.001:
                mode = "arrest"
                quiet = 0
            else:
                command = min(open_limit, command + 0.006)
                if speed > 0.015:
                    mode = "slide"
        if mode == "arrest":
            command = hold_command
            quiet = quiet + 1 if speed < 0.001 else 0
            if quiet >= 4:
                if displacement >= target - tolerance + 0.0005:
                    mode = "hold"
                    held = 0
                else:
                    mode = "unload"
                    quiet = 0
        elif mode == "hold":
            command = hold_command
            held += 1
            if speed > 0.005:
                mode = "arrest"
                quiet = 0
            elif held > math.ceil(3.0 / env.step_dt):
                raise RuntimeError("Pulsed slip failed the native stable hold")
        env.action[:, -1] = command
        _step(env, motion)

    raise RuntimeError("Pulsed slip did not complete within its physical episode budget")


def execute_contact(env, motion):
    """Attempt one contact task using shared geometry paths and robot actions."""
    from benchmark.tasks.extra.interaction import interaction_path, path_quaternions, _rotation

    kind = env.task["episode_config"]["task_type"]
    if kind not in {"push", "roll", "gears", "hang", "slip"}:
        raise ValueError(f"No contact expert for {kind}")
    _state(env)
    path = interaction_path(env)
    quaternions = path_quaternions(env, path)
    if len(path) < 3 or len(quaternions) != len(path):
        raise ValueError("Contact expert requires a complete shared path and matching orientations")
    if kind in {"push", "roll"}:
        layout = _rotation(env.reset_poses["mechanism"]["quaternion_wxyz"])
        (_roll_two_pad if kind == "roll" else _surface)(env, motion, path, quaternions, layout)
    else:
        {"gears": _gears, "hang": _hang, "slip": _slip}[kind](env, motion, path, quaternions)
