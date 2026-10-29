"""Physics-rate history checks for Extra's five contact manipulation tasks.

Samples are native measurements relative to settled reset poses. Slip uses a
separate reference established after a stable, unsupported two-pad grasp.
This module neither imports the simulator nor controls task bodies.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import math
from numbers import Real


CONTACT_KINDS = frozenset(("push", "roll", "gears", "hang", "slip"))
COMMON_FIELDS = ("gel_force_n", "object_speed_m_s", "angular_speed_rad_s")
SAMPLE_FIELDS = {
    "push": ("path_u", "cross_track_m", "table_force_n", "lift_m",
             "gel_total_force_n", "housing_total_force_n"),
    "roll": ("roll_angle_rad", "translation_m", "slip_ratio", "table_force_n", "off_axis_rad",
             "lift_m", "table_clearance_m", "rolling_arc_length_m",
             "gel_total_force_n", "housing_total_force_n"),
    "gears": ("input_angle_rad", "output_fast_angle_rad", "output_slow_angle_rad",
              "pair_fast_force_n", "pair_slow_force_n", "robot_output_force_n", "other_robot_input_force_n",
              "gel_total_force_n", "housing_total_force_n"),
    "hang": ("hook_through_handle", "handle_above_hook", "lowering_m", "hook_up_force_n",
             "other_support_force_n", "robot_force_n", "release_clearance_m",
             "gel_total_force_n", "housing_total_force_n"),
    "slip": ("slip_armed", "axial_slip_m", "axial_speed_m_s", "hand_translation_m",
             "hand_rotation_rad", "relative_rotation_rad", "rod_axis_tilt_rad", "left_gel_force_n", "right_gel_force_n",
             "left_gel_contact_present", "right_gel_contact_present",
             "housing_force_n", "housing_contact_load_n", "gel_up_force_n", "housing_up_force_n",
             "external_support_force_n", "suspended", "gripper_command", "gripper_aperture_m"),
}
BOOLEAN_FIELDS = frozenset(("hook_through_handle", "handle_above_hook", "slip_armed", "suspended",
                            "left_gel_contact_present", "right_gel_contact_present"))


@lru_cache(maxsize=32)
def _push_arc_table(length, amplitude):
    """Composite Simpson integration, cached for the fixed task geometry."""
    count, total = 1024, 0.
    h = 1/count
    def speed(u):
        return math.hypot(length, 2*math.pi*amplitude*math.cos(2*math.pi*u))
    values = [0.]
    for index in range(count):
        total += h/6*(speed(index*h)+4*speed((index+.5)*h)+speed((index+1)*h))
        values.append(total)
    return tuple(values)


def push_arc_length(u, goal):
    """True S-curve length from parameter zero to u; the X span stays unchanged."""
    length, amplitude = float(goal["length_m"]), float(goal["amplitude_m"])
    if (not isinstance(u, Real) or isinstance(u, bool) or not math.isfinite(u) or not 0 <= u <= 1
            or not math.isfinite(length) or length <= 0 or not math.isfinite(amplitude)):
        raise ValueError("Push arc length needs finite geometry and u in [0, 1]")
    table = _push_arc_table(length, amplitude)
    location = u*(len(table)-1)
    index = min(int(location), len(table)-2)
    return table[index]+(location-index)*(table[index+1]-table[index])


def cylinder_minimum_z(center_z, axis_z, radius, length):
    """Exact vertical support of a finite cylinder, independent of axial roll."""
    if (not all(math.isfinite(v) for v in (center_z, axis_z, radius, length))
            or radius <= 0 or length <= 0 or abs(axis_z) > 1.0 + 1e-12):
        raise ValueError("Cylinder support needs finite geometry and a normalized axis")
    axis_z = min(1.0, abs(axis_z))
    return center_z - radius * math.sqrt(max(0.0, 1.0 - axis_z**2)) - length * axis_z / 2


def _finite_values(value, name):
    if isinstance(value, Real) and not math.isfinite(value):
        raise RuntimeError(f"Non-finite contact state: {name}")
    if isinstance(value, dict):
        for key, child in value.items():
            _finite_values(child, f"{name}/{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_values(child, f"{name}/{index}")


class ContactProgress:
    """Fail closed on incomplete traces, lost load paths and unsupported shortcuts."""

    def __init__(self, task):
        self.task = task
        self.kind = task["episode_config"]["task_type"]
        if self.kind not in CONTACT_KINDS:
            raise ValueError(f"Unknown contact task type: {self.kind}")
        self.goal, self.config = task["goals"][0], task["success_config"]
        self.reset()

    def reset(self):
        """Clear every episode history; references remain owned by the native runtime."""
        self.last_sample = None
        self.previous = None
        self.samples = self.motion_samples = self.gel_samples = self.support_samples = 0
        self.pair_samples = 0
        self.elapsed = self.trace_time = self.stable_time = 0.0
        self.active_samples = self.active_gel_samples = self.active_support_samples = 0
        self.gel_contact_gap_s = self.table_contact_gap_s = 0.
        self.contact_seen = self.active = False
        self.invalid_reason = None
        self.covered_u = 0.0
        self.furthest_u = 0.0
        self.covered_intervals = []
        self.covered_arc_length_m = 0.
        self.path_arc_length_m = push_arc_length(1., self.goal) if self.kind == "push" else 0.
        self.above_seen = self.threaded_seen = False
        self.slip_armed = self.slip_observed = self.held_seen = False
        self.max_slip_m = 0.0
        self.housing_support_fraction = self.gel_weight_fraction = 0.0
        self.sliding_open_command = self.sliding_open_aperture_m = None
        self.tightening_seen = False
        self.result = {"success": False, "reason": "native contact history has not started"}

    def _invalidate(self, reason):
        if self.invalid_reason is None:
            self.invalid_reason = reason

    def _validate(self, sample):
        for key in (*COMMON_FIELDS, *SAMPLE_FIELDS[self.kind]):
            if key not in sample:
                raise ValueError(f"Missing required {self.kind} contact state: {key}")
            value = sample[key]
            if key in BOOLEAN_FIELDS:
                if not isinstance(value, (bool, Real)) or value not in (False, True):
                    raise ValueError(f"Contact state {key} must be boolean")
            elif not isinstance(value, Real) or isinstance(value, bool):
                raise ValueError(f"Contact state {key} must be a finite scalar")
        _finite_values(sample, self.kind)
        for key in (*COMMON_FIELDS, *SAMPLE_FIELDS[self.kind]):
            if (key.endswith("force_n") or key in {"object_speed_m_s", "angular_speed_rad_s",
                                                   "cross_track_m", "slip_ratio", "off_axis_rad",
                                                   "rolling_arc_length_m",
                                                   "hand_translation_m", "hand_rotation_rad",
                                                   "relative_rotation_rad", "rod_axis_tilt_rad"}) and sample[key] < 0:
                raise ValueError(f"Contact state {key} must be nonnegative")
        if self.kind == "slip":
            if not -1 <= sample["gripper_command"] <= 1:
                raise ValueError("Contact state gripper_command must be in [-1, 1]")
            if sample["gripper_aperture_m"] < 0 or sample["housing_contact_load_n"] < 0:
                raise ValueError("Gripper aperture and housing contact load must be nonnegative")

    def update(self, sample, dt, *, sample_id=None, advanced=True):
        if not isinstance(dt, Real) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Contact progress dt must be positive and finite")
        if dt > self.task["episode_config"]["physics_dt"] + 1e-12:
            raise ValueError("Contact history must be sampled at the physics rate")
        if not advanced:
            return deepcopy(self.result)
        if type(sample_id) is not int or sample_id < 1:
            raise ValueError("Contact sample IDs must be positive integers")
        if sample_id == self.last_sample:
            return deepcopy(self.result)
        if self.last_sample is not None and sample_id != self.last_sample+1:
            raise ValueError("Contact sample IDs must increase monotonically without gaps")
        self._validate(sample)
        self.last_sample = sample_id
        self.samples += 1
        self.elapsed += dt
        if self.elapsed > self.task["episode_config"]["timeout_s"]+1e-12:
            self._invalidate("contact episode timed out")
        contact = sample["gel_force_n"] >= self.config["contact_threshold_n"]
        self.contact_seen |= contact
        ready = getattr(self, f"_{self.kind}")(sample, dt, contact)
        if self.kind != "slip" and (self.active or self.kind == "hang"):
            # Hook support is separate; only native robot-target reactions count here.
            housing, gel = sample["housing_total_force_n"], sample["gel_total_force_n"]
            if housing > .05 and housing / (gel + housing) > .1:
                self._invalidate("sensor housing dominated the manipulation load")
        quiet = (sample["object_speed_m_s"] <= self.config["max_object_speed_m_s"]
                 and sample["angular_speed_rad_s"] <= self.config["max_angular_speed_rad_s"])
        ready = ready and quiet and self.contact_seen and self.invalid_reason is None
        self.stable_time = self.stable_time + dt if ready else 0.0
        history = {"physics_sample_count": self.samples, "elapsed_s": self.elapsed,
                   "motion_sample_count": self.motion_samples, "trace_duration_s": self.trace_time,
                   "invalid_reason": self.invalid_reason}
        if self.kind in {"push", "roll"}:
            history.update(gel_contact_fraction=self.gel_samples / max(1, self.motion_samples),
                           table_support_fraction=self.support_samples / max(1, self.motion_samples))
        if self.kind == "push":
            history.update(path_coverage=self.covered_arc_length_m/self.path_arc_length_m,
                           parameter_coverage=self.covered_u,
                           path_arc_length_m=self.path_arc_length_m,
                           covered_arc_length_m=self.covered_arc_length_m,
                           arc_coverage=self.covered_arc_length_m/self.path_arc_length_m)
        if self.kind == "gears":
            history["pair_contact_fraction"] = self.pair_samples / max(1, self.motion_samples)
            history["input_gel_contact_fraction"] = self.gel_samples / max(1, self.motion_samples)
        if self.kind in {"push", "roll", "gears"}:
            history.update(active_sample_count=self.active_samples,
                           active_gel_contact_fraction=self.active_gel_samples/max(1, self.active_samples),
                           gel_contact_gap_s=self.gel_contact_gap_s)
            if self.kind in {"push", "roll"}:
                history.update(active_table_support_fraction=self.active_support_samples/max(1, self.active_samples),
                               table_contact_gap_s=self.table_contact_gap_s)
        if self.kind == "hang":
            history.update(above_seen=self.above_seen, threaded_seen=self.threaded_seen)
        if self.kind == "slip":
            history.update(slip_armed=self.slip_armed, slip_observed=self.slip_observed,
                           held_seen=self.held_seen, maximum_axial_slip_m=self.max_slip_m,
                           housing_support_fraction=self.housing_support_fraction,
                           gel_weight_fraction=self.gel_weight_fraction,
                           sliding_open_command=self.sliding_open_command,
                           sliding_open_aperture_m=self.sliding_open_aperture_m,
                           tightening_seen=self.tightening_seen)
        success = ready and self.stable_time + 1e-12 >= self.config["stable_duration_s"]
        self.result = {"success": success, "contact_seen": self.contact_seen,
                       "stable_duration_s": self.stable_time, "metrics": deepcopy(sample), "history": history}
        if not success:
            self.result["reason"] = self.invalid_reason or "contact trajectory or stability window incomplete"
        self.previous = dict(sample)
        return deepcopy(self.result)

    def _motion_contact(self, sample, contact, moving, dt):
        if moving:
            self.motion_samples += 1
            self.trace_time += dt
            self.gel_samples += int(contact)
            self.support_samples += int(sample["table_force_n"] >= self.config["contact_threshold_n"])
        return (self.motion_samples > 0
                and self.gel_samples / self.motion_samples + 1e-12 >= self.goal["contact_fraction"]
                and self.support_samples / self.motion_samples + 1e-12 >= self.goal["support_fraction"])

    def _active_contact(self, sample, contact, dt, *, table=False):
        if not self.active:
            return
        self.active_samples += 1
        self.active_gel_samples += int(contact)
        self.gel_contact_gap_s = 0. if contact else self.gel_contact_gap_s+dt
        if self.gel_contact_gap_s > self.goal.get("max_gel_contact_gap_s", .01)+1e-12:
            self._invalidate("gel contact was lost during active manipulation")
        if table:
            supported = sample["table_force_n"] >= self.config["contact_threshold_n"]
            self.active_support_samples += int(supported)
            self.table_contact_gap_s = 0. if supported else self.table_contact_gap_s+dt
            if self.table_contact_gap_s > self.goal.get("max_table_contact_gap_s", .01)+1e-12:
                self._invalidate("rolling lost tabletop support during active manipulation")

    def _push(self, sample, dt, contact):
        goal, u = self.goal, sample["path_u"]
        if self.previous is None and abs(u) > 0.02:
            self._invalidate("push trace did not begin at its reset position")
        prior = self.previous["path_u"] if self.previous is not None else u
        # Stationary normal preload may cross the force threshold repeatedly.
        # Start manipulation continuity at 0.1 mm of measured object travel.
        self.active |= max(abs(u)*goal["length_m"],sample["cross_track_m"]) > .0001
        self._active_contact(sample, contact, dt, table=True)
        moving = self.active and abs(u - prior) > 1e-8 and sample["object_speed_m_s"] > 0.0001
        valid_path = (sample["cross_track_m"] <= goal["cross_track_m"] and -0.02 <= u <= 1.02)
        if sample["lift_m"] > goal["max_lift_m"]:
            self._invalidate("push object was lifted instead of supported")
        if self.active:
            if not valid_path:
                self._invalidate("push object left its prescribed path")
            if (u - prior) * goal["length_m"] > goal["max_path_speed_m_s"] * dt + 1e-5:
                self._invalidate("push path jumped over unobserved contact history")
            if u < self.furthest_u - 0.02:
                self._invalidate("push path was traversed out of order")
            self.furthest_u = max(self.furthest_u, u)
            # Coverage is earned only while the gel and table support the moving block.
            if (moving and min(1., u) > max(0., prior) and contact
                    and sample["table_force_n"] >= self.config["contact_threshold_n"]
                    and valid_path):
                intervals = sorted([*self.covered_intervals, (max(0.0, prior), min(1.0, u))])
                merged = []
                for lower, upper in intervals:
                    if merged and lower <= merged[-1][1] + 1e-12:
                        merged[-1] = (merged[-1][0], max(merged[-1][1], upper))
                    else:
                        merged.append((lower, upper))
                self.covered_intervals = merged
                self.covered_u = sum(upper - lower for lower, upper in merged)
                self.covered_arc_length_m = sum(push_arc_length(upper, goal)-push_arc_length(lower, goal)
                                               for lower, upper in merged)
        fractions = self._motion_contact(sample, contact, moving, dt)
        return (self.covered_arc_length_m/self.path_arc_length_m + 1e-12 >= goal["coverage"] and u >= goal["coverage"]
                and fractions and self.trace_time + 1e-12 >= goal["minimum_duration_s"]
                and valid_path and contact and sample["table_force_n"] >= self.config["contact_threshold_n"])

    def _roll(self, sample, dt, contact):
        goal, angle, x = self.goal, sample["roll_angle_rad"], sample["translation_m"]
        if self.previous is None and (abs(angle) > 0.02 or abs(x) > 0.001):
            self._invalidate("rolling trace did not begin at its reset position")
        prior = self.previous["roll_angle_rad"] if self.previous is not None else angle
        # Table settling before the first gel contact is not a rolling
        # manipulation; only motion while the robot actually loads the can is
        # checked against the rolling gates.
        self.active |= contact and max(abs(angle)*goal["radius_m"],abs(x)) > .0001
        self._active_contact(sample, contact, dt, table=True)
        moving = self.active and abs(angle - prior) > 1e-8 and sample["angular_speed_rad_s"] > 0.001
        if (sample["table_clearance_m"] > goal.get("max_table_clearance_m", .003)+1e-12
                or sample["table_clearance_m"] < -goal.get("max_table_penetration_m", .001)-1e-12):
            self._invalidate("rolling geometry left or penetrated the tabletop")
        if self.active:
            # The pad can already have rolled the can through its approach
            # before the .05 N preload arms tracking. The first tracked sample
            # then divides one transient by a near-zero arc; evaluate the ratio
            # only after the tracked rolling arc is measurable.
            if (sample["rolling_arc_length_m"] >= .0005
                    and sample["slip_ratio"] > goal["max_slip_ratio"]):
                self._invalidate("rolling exceeded its slip limit")
            if sample["off_axis_rad"] > goal["max_off_axis_rad"]:
                self._invalidate("rolling left its prescribed axis")
            if abs(x - goal["radius_m"] * angle) > goal["translation_tolerance_m"]:
                self._invalidate("rolling translation is inconsistent with radius times angle")
            if abs(angle - prior) > 0.04:
                self._invalidate("rolling angle jumped over unobserved physics history")
        fractions = self._motion_contact(sample, contact, moving, dt)
        return (abs(angle - goal["target_rad"]) <= goal["tolerance_rad"]
                and abs(x - goal["radius_m"] * angle) <= goal["translation_tolerance_m"]
                and fractions and contact and sample["table_force_n"] >= self.config["contact_threshold_n"])

    def _gears(self, sample, dt, contact):
        goal = self.goal
        angles = (sample["input_angle_rad"], sample["output_fast_angle_rad"], sample["output_slow_angle_rad"])
        if self.previous is None and max(abs(value) for value in angles) > 0.02:
            self._invalidate("gear trace did not begin at its reset angles")
        prior = self.previous["input_angle_rad"] if self.previous is not None else angles[0]
        moving = abs(angles[0] - prior) > 1e-8 and sample["angular_speed_rad_s"] > 0.001
        # One-pad contact under preload can rock the train through its tooth
        # backlash (a few mrad) before the grasp exists. Arm the manipulation
        # continuity gate only once the input turns measurably.
        self.active |= max(abs(value) for value in angles) > .01
        self._active_contact(sample, contact, dt)
        if sample["robot_output_force_n"] >= goal["max_robot_output_force_n"]:
            self._invalidate("robot directly contacted an output gear")
        if sample["other_robot_input_force_n"] >= goal.get("max_other_robot_input_force_n", .001):
            self._invalidate("other robot bodies drove the input gear")
        if self.active:
            if (abs(angles[1] - goal["fast_ratio"] * angles[0]) > goal["ratio_tolerance_rad"]
                    or abs(angles[2] - goal["slow_ratio"] * angles[0]) > goal["ratio_tolerance_rad"]):
                self._invalidate("gear trajectory violated the transmission ratio")
            if abs(angles[0] - prior) > 0.04:
                self._invalidate("gear angle jumped over unobserved physics history")
        if moving:
            self.motion_samples += 1
            self.trace_time += dt
            self.gel_samples += int(contact)
            self.pair_samples += int(min(sample["pair_fast_force_n"], sample["pair_slow_force_n"])
                                     >= goal["pair_contact_threshold_n"])
        return (abs(angles[0] - goal["input_target_rad"]) <= goal["input_tolerance_rad"]
                and abs(angles[1] - goal["output_fast_target_rad"]) <= goal["output_fast_tolerance_rad"]
                and abs(angles[2] - goal["output_slow_target_rad"]) <= goal["output_slow_tolerance_rad"]
                and self.motion_samples > 0
                and contact
                and self.gel_samples/self.motion_samples + 1e-12 >= goal.get("input_gel_contact_fraction", .95)
                and self.pair_samples / self.motion_samples + 1e-12 >= goal["pair_contact_fraction"])

    def _hang(self, sample, dt, contact):
        goal = self.goal
        was_above = self.above_seen
        self.above_seen |= contact and bool(sample["handle_above_hook"])
        self.threaded_seen |= was_above and bool(sample["hook_through_handle"])
        weight = self.task["objects"]["mechanism"]["mass_kg"] * 9.81
        return (self.threaded_seen and sample["hook_through_handle"]
                and sample["lowering_m"] >= goal["minimum_lowering_m"]
                and sample["robot_force_n"] < goal["release_force_n"]
                and sample["release_clearance_m"] >= goal.get("minimum_release_clearance_m", .01)
                and sample["other_support_force_n"] < goal["max_other_support_force_n"]
                and goal["min_hook_weight_fraction"] * weight <= sample["hook_up_force_n"]
                <= goal["max_hook_weight_fraction"] * weight)

    def _slip(self, sample, dt, contact):
        goal = self.goal
        if not self.slip_armed:
            if not sample["slip_armed"]:
                return False
            self.slip_armed = True
            if abs(sample["axial_slip_m"]) > 0.001:
                self._invalidate("slip reference was not initialized before the slide")
        if not sample["slip_armed"]:
            self._invalidate("slip grasp reference was lost")
        threshold = self.config["contact_threshold_n"]
        weight = self.task["objects"]["mechanism"]["mass_kg"] * 9.81
        self.housing_support_fraction = sample["housing_up_force_n"] / max(
            sample["gel_up_force_n"] + sample["housing_up_force_n"], 1e-12)
        self.gel_weight_fraction = sample["gel_up_force_n"] / weight
        if not sample["suspended"] or sample["external_support_force_n"] >= goal["max_external_support_force_n"]:
            self._invalidate("slip object acquired external support")
        if not (sample["left_gel_contact_present"] and sample["right_gel_contact_present"]):
            self._invalidate("slip object was dropped or regrasped")
        if self.housing_support_fraction > goal["max_housing_force_fraction"] + 1e-12:
            self._invalidate("slip object was supported by the sensor housing")
        if sample["housing_contact_load_n"] > goal.get("max_housing_contact_force_n", .05)+1e-12:
            self._invalidate("slip object acquired mechanical sensor housing contact")
        if (sample["hand_translation_m"] > goal["max_hand_translation_m"]
                or sample["hand_rotation_rad"] > goal["max_hand_rotation_rad"]
                or sample["rod_axis_tilt_rad"] > goal["max_axis_tilt_rad"]):
            self._invalidate("slip was replaced by hand motion or object rotation")
        slip = sample["axial_slip_m"]
        if self.previous is not None and self.previous["slip_armed"]:
            maximum_step = max(0.002, 2 * abs(sample["axial_speed_m_s"]) * dt + 0.0005)
            if abs(slip - self.previous["axial_slip_m"]) > maximum_step:
                self._invalidate("slip jumped over unobserved axial motion")
        self.max_slip_m = max(self.max_slip_m, slip)
        if slip > goal["target_slip_m"] + goal["tolerance_m"] or slip < -0.001:
            self._invalidate("slip exceeded its permitted axial interval")
        if slip < self.max_slip_m - 0.001:
            self._invalidate("slip object was pulled back instead of arrested")
        sliding = slip >= .002 and sample["axial_speed_m_s"] > goal["max_axial_speed_m_s"]
        self.slip_observed |= sliding
        if sliding:
            command, aperture = sample["gripper_command"], sample["gripper_aperture_m"]
            self.sliding_open_command = command if self.sliding_open_command is None else max(self.sliding_open_command, command)
            self.sliding_open_aperture_m = aperture if self.sliding_open_aperture_m is None else max(self.sliding_open_aperture_m, aperture)
        tightened = (self.sliding_open_command is not None
            and self.sliding_open_command-sample["gripper_command"] + 1e-12 >= goal.get("min_gripper_command_closure", .001)
            and self.sliding_open_aperture_m-sample["gripper_aperture_m"] + 1e-12 >= goal.get("min_gripper_aperture_closure_m", .00002))
        self.tightening_seen |= tightened
        ready = (self.slip_observed and abs(slip - goal["target_slip_m"]) <= goal["tolerance_m"]
                 and abs(sample["axial_speed_m_s"]) <= goal["max_axial_speed_m_s"]
                 and min(sample["left_gel_force_n"], sample["right_gel_force_n"]) >= threshold
                 and self.gel_weight_fraction + 1e-12 >= goal["min_gel_weight_fraction"]
                 and tightened)
        self.held_seen |= ready and self.invalid_reason is None
        return ready
