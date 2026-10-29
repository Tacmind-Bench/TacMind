"""Extra suite layouts, passive mechanisms and GSmini reset/step environment.

Catalog, reset sampling and progress checks import without Isaac. Construct
ExtraEnvironment only after AppLauncher(enable_cameras=True).
"""

from __future__ import annotations


from copy import deepcopy
import math
import random


def sample_extra_reset(task, seed, *, reachable=None, max_attempts=32):
    """Sample one common SE(2) transform, preserving every assembly offset."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Extra seed must be an integer in [0, 2**32)")
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    rng = random.Random(seed)
    contract = task["reset_contract"]
    widths = contract["xy_jitter_m"]
    yaw_width = contract["yaw_jitter_rad"]
    if len(widths) != 2 or any(not math.isfinite(x) or x < 0 for x in (*widths, yaw_width)):
        raise ValueError("Extra reset jitter must be finite and nonnegative")
    anchor = contract.get("anchor_m", task["objects"][task["obj_of_interest"][0]]["initial_position"])
    for _ in range(max_attempts):
        dx, dy = (rng.uniform(-x, x) for x in widths)
        yaw = rng.uniform(-yaw_width, yaw_width)
        c, s = math.cos(yaw), math.sin(yaw)
        poses = {}
        for name, spec in task["objects"].items():
            x, y, z = spec["initial_position"]
            theta = spec.get("initial_yaw_rad", 0.0) + yaw
            q = [math.cos(theta/2), 0.0, 0.0, math.sin(theta/2)]
            if "initial_quat" in spec:
                qw, qx, qy, qz = spec["initial_quat"]
                c0, s0 = math.cos(yaw/2), math.sin(yaw/2)
                q = [c0*qw-s0*qz, c0*qx-s0*qy, c0*qy+s0*qx, c0*qz+s0*qw]
            joints = dict(spec.get("joint_defaults", {}))
            for joint, bounds in spec.get("joint_position_ranges", {}).items():
                if len(bounds) != 2 or any(not math.isfinite(v) for v in bounds) or bounds[0] > bounds[1]:
                    raise ValueError(f"Invalid Extra joint reset range: {joint}")
                joints[joint] = rng.uniform(*bounds)
            poses[name] = {"position": [anchor[0]+dx+c*(x-anchor[0])-s*(y-anchor[1]),
                                         anchor[1]+dy+s*(x-anchor[0])+c*(y-anchor[1]), z],
                           "quaternion_wxyz": q, "joint_positions": joints}
            if spec.get("reset_frame") == "robot_hand":
                poses[name] = {"position": list(spec["initial_position"]),
                               "quaternion_wxyz": list(spec["initial_quat"]), "joint_positions": joints}
        if task["episode_config"]["task_type"] == "surface_inspection":
            from benchmark.tasks.extra.inspection import sample_slots
            sample_slots(poses, rng)
        from benchmark.tasks.extra.visuals import layout_frame_report
        if not layout_frame_report(task, poses)["passed"]:
            continue
        if reachable is None or reachable(poses):
            return poses
    raise RuntimeError(f"No reachable Extra layout inside AgentView after {max_attempts} attempts")


def extra_success_contract(task, backend="gsmini"):
    from benchmark.catalog import is_extra_scene
    if is_extra_scene(task):
        from benchmark.tasks.extra.scenes import extra_scene_success_contract
        return extra_scene_success_contract(task, backend)
    if backend != "gsmini":
        raise ValueError("Extra environments currently support GSmini only")
    contract = {"format": "tacmind.extra.mechanism-goals", "suite": "tac_extra", "backend": backend,
            "task_id": task["task_id"], "frame": "reset_mechanism",
            "pose_window_s": 0.0, "stable_steps": math.ceil(task["success_config"]["stable_duration_s"] / 0.05),
            "goal": deepcopy(task["goals"][0]), **deepcopy(task["success_config"])}
    if task["episode_config"]["task_type"] == "slice":
        contract.update(format="tacmind.extra.cutting-goals", frame="live_food_and_board",
                        cutting_model=task["episode_config"]["cutting_model"])
    if task["episode_config"]["task_type"] in {"drawer", "door"}:
        contract["format"] = "tacmind.extra.cabinet-goals"
    from benchmark.tasks.extra.contact import CONTACT_KINDS
    if task["episode_config"]["task_type"] in CONTACT_KINDS:
        contract.update(format="tacmind.extra.contact-goals", sampling="native physics steps",
                        physics_dt=task["episode_config"]["physics_dt"])
    if task["episode_config"]["task_type"] in {"stand", "press_slide", "surface_inspection"}:
        contract.update(format=("tacmind.extra.standing-goals" if task["episode_config"]["task_type"] == "stand"
                                 else "tacmind.extra.press-slide-goals"), sampling="native physics steps",
                        physics_dt=task["episode_config"]["physics_dt"])
    if task["episode_config"]["task_type"] == "surface_inspection":
        contract["format"] = "tacmind.extra.surface-inspection-goals"
    if task["episode_config"]["task_type"] in {"wipe_bottle", "wipe_whiteboard"}:
        contract.update(format="tacmind.extra.wiping-goals", frame="live_surface",
                        sampling="native physics steps", physics_dt=task["episode_config"]["physics_dt"])
    if task["episode_config"]["task_type"] == "ram":
        goal_format = "tacmind.extra.ram-indentation-limited-goals" if "max_gel_indentation_m" in task["goals"][0] else "tacmind.extra.ram-contact-goals"
        contract.update(format=goal_format, frame="live_socket",
                        sampling="native physics steps", physics_dt=task["episode_config"]["physics_dt"])
    return contract


def switch_effort(position, velocity, *, well_rad, potential_coefficient, damping_nm_s):
    """-dU/dq - c*qdot, U=k/4*(q²-a²)². No goal-conditioned actuation."""
    return -potential_coefficient * position * (position**2-well_rad**2) - damping_nm_s * velocity


class ExtraProgress:
    """Pure, per-episode history; simulator truth is diagnostic input, never an action."""

    def __new__(cls, task):
        if task["episode_config"]["task_type"] in {"wipe_bottle", "wipe_whiteboard"}:
            from benchmark.tasks.extra.wiping import WipingProgress
            return WipingProgress(task)
        if task["episode_config"]["task_type"] == "ram":
            from benchmark.tasks.extra.assembly import RamProgress
            return RamProgress(task)
        if task["episode_config"]["task_type"] == "surface_inspection":
            from benchmark.tasks.extra.inspection import InspectionProgress
            return InspectionProgress(task)
        if task["episode_config"]["task_type"] == "press_slide":
            from benchmark.tasks.extra.latch import LatchProgress
            return LatchProgress(task)
        if task["episode_config"]["task_type"] == "stand":
            from benchmark.tasks.extra.placement import StandProgress
            return StandProgress(task)
        from benchmark.tasks.extra.contact import CONTACT_KINDS, ContactProgress
        if task["episode_config"]["task_type"] in CONTACT_KINDS:
            return ContactProgress(task)
        return super().__new__(cls)

    def __init__(self, task):
        self.task = task
        self.goal = task["goals"][0]
        self.kind = task["episode_config"]["task_type"]
        self.config = task["success_config"]
        self.last_sample = None
        self.contact_seen = False
        self.pressed = False
        self.press_time = self.stable_time = 0.0
        self.trace_frames = self.contact_frames = self.support_frames = 0
        self.next_bin = 0
        self.trace_time = self.contact_gap = 0.0
        self.crossed = False
        self.support_gap_s = 0.0
        self.lost_support = False
        self.grip_reference_xyzw = None
        self.cabinet_attachment_lost = False
        self.tool_grasp_lost = False
        self.result = {"success": False, "reason": "no physical interaction observed"}

    def update(self, sample, dt, *, sample_id=None, advanced=True):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Extra progress dt must be positive and finite")
        if not advanced or (sample_id is not None and sample_id == self.last_sample):
            return dict(self.result)
        self.last_sample = sample_id
        for key, value in sample.items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise RuntimeError(f"Non-finite Extra state: {key}")
        contact = sample.get("gel_force_n", 0.0) >= self.config["contact_threshold_n"]
        self.contact_seen |= contact
        two_pads = all(sample.get(f"{side}_gel_force_n", 0.0) >= self.config["contact_threshold_n"]
                       for side in ("left", "right"))
        if self.kind == "pivot" and two_pads and self.grip_reference_xyzw is None:
            self.grip_reference_xyzw = sample.get("grip_rotation_xyzw")
        quiet = abs(sample.get("joint_speed", 0.0)) <= self.config["max_joint_speed"]
        if self.kind in {"drawer", "door"}:
            self.cabinet_attachment_lost |= not bool(sample.get("cabinet_intact", False))
        ready = False
        goal = self.goal
        if self.kind == "drawer":
            ready = sample["joint_position"] >= goal["minimum_m"] and quiet
        elif self.kind == "door":
            ready = sample["joint_displacement"] >= goal["minimum_rad"] and quiet
        elif self.kind == "valve":
            ready = abs(sample["joint_displacement"]-goal["target_rad"]) <= goal["tolerance_rad"] and quiet
        elif self.kind == "button":
            self.press_time = self.press_time+dt if contact and sample["depth_m"] >= goal["press_depth_m"] else 0.0
            self.pressed |= self.press_time + 1e-9 >= goal["press_duration_s"]
            ready = self.pressed and sample["depth_m"] <= goal["release_depth_m"] and sample.get("robot_force_n", 0.0) < 0.05 and quiet
        elif self.kind == "switch":
            self.crossed |= self.contact_seen and sample["joint_position"] > 0.0
            ready = self.crossed and abs(sample["joint_position"]-goal["target_rad"]) <= goal["tolerance_rad"] and sample.get("robot_force_n", 0.0) < 0.05 and quiet
        elif self.kind == "cap":
            ready = (sample["withdrawal_m"] >= goal["minimum_m"] and sample["clearance_m"] >= goal["clearance_m"]
                     and sample.get("object_speed_m_s", 0.0) < 0.01 and sample.get("holder_force_n", 0.0) < 0.05)
        elif self.kind == "contour":
            index = int(sample.get("path_bin", -1))
            on_path = contact and sample.get("cross_track_m", math.inf) <= goal["cross_track_m"]
            if self.next_bin or (on_path and index == 0):
                self.contact_gap = 0.0 if on_path else self.contact_gap+dt
                # At 20 Hz a continuous contact patch can move across two
                # adjacent 4 mm bins between observations; large jumps do not
                # count as traversal. Preparation at the start is not tracing.
                if on_path and self.next_bin <= index <= self.next_bin+1:
                    self.next_bin = index+1
                if self.next_bin >= 2:
                    self.trace_frames += 1
                    self.trace_time += dt
                    self.contact_frames += int(on_path)
                if self.contact_gap > goal["max_contact_gap_s"]:
                    self.next_bin = self.trace_frames = self.contact_frames = 0
                    self.trace_time = self.contact_gap = 0.0
            ready = (self.next_bin/goal["bins"] >= goal["coverage"]
                     and self.contact_frames/max(1, self.trace_frames) >= goal["contact_fraction"]
                     and self.trace_time >= goal["minimum_duration_s"] and on_path)
        elif self.kind == "pivot":
            if self.contact_seen:
                self.trace_frames += 1
                supported = sample.get("table_force_n", 0.0) >= 0.05
                self.support_frames += int(supported)
                self.support_gap_s = 0.0 if supported else self.support_gap_s+dt
                self.lost_support |= self.support_gap_s > goal.get("max_support_gap_s", 0.15)+1e-9
            ready = (abs(sample["pivot_rad"]-goal["target_rad"]) <= goal["tolerance_rad"]
                     and sample["relative_rotation_rad"] >= goal["relative_rotation_min_rad"]
                     and not self.lost_support
                     and sample.get("off_axis_rad", 0.0) <= goal["tolerance_rad"]
                     and sample.get("table_force_n", 0.0) >= 0.05
                     and self.support_frames/max(1, self.trace_frames) >= goal["support_fraction"]
                     and sample.get("object_speed_m_s", 0.0) < 0.01)
        elif self.kind == "slice":
            self.tool_grasp_lost |= not bool(sample["tool_grasped"])
            separated = (sample["pieces_count"] == goal["target_pieces"] if "target_pieces" in goal
                         else sample["separation_m"] >= goal["minimum_separation_m"])
            ready = (sample["target_plane_released"] and sample["cut_count"] == goal["required_cuts"]
                     and separated
                     and sample["all_pieces_on_board"] and sample["tool_lifted"] and sample["tool_grasped"]
                     and sample["knife_clearance_m"] >= goal["knife_clearance_m"]
                     and sample["max_piece_speed_m_s"] <= goal["max_piece_speed_m_s"]
                     and not self.tool_grasp_lost)
        else:
            raise ValueError(f"Unknown Extra task type: {self.kind}")
        ready = ready and self.contact_seen
        if self.kind in {"drawer", "door"}:
            ready = ready and not self.cabinet_attachment_lost
        self.stable_time = self.stable_time+dt if ready else 0.0
        duration = goal.get("release_duration_s", self.config["stable_duration_s"])
        success = ready and (self.kind == "contour" or self.stable_time+1e-9 >= duration)
        self.result = {"success": success, "contact_seen": self.contact_seen,
                       "stable_duration_s": self.stable_time, "metrics": dict(sample)}
        if self.kind in {"drawer", "door"}:
            self.result["history"] = {"cabinet_attachment_lost": self.cabinet_attachment_lost}
        if self.kind == "slice":
            self.result["history"] = {"tool_grasp_lost": self.tool_grasp_lost}
        if self.kind == "contour":
            self.result["history"] = {"covered_bins": self.next_bin, "total_bins": goal["bins"],
                                      "contact_fraction": self.contact_frames/max(1,self.trace_frames)}
        elif self.kind == "pivot":
            self.result["history"] = {"support_fraction": self.support_frames/max(1,self.trace_frames),
                                      "lost_support": self.lost_support}
        if not success:
            self.result["reason"] = "task history or stability window incomplete"
        return dict(self.result)


class ExtraSuccessMonitor:
    """Adapter for the shared TaskSuccessMonitor interface."""

    def __new__(cls, env, task, backend="gsmini"):
        from benchmark.catalog import is_extra_scene
        if is_extra_scene(task):
            from benchmark.tasks.extra.scenes import ExtraSceneSuccessMonitor
            return ExtraSceneSuccessMonitor(env, task, backend)
        return super().__new__(cls)

    def __init__(self, env, task, backend="gsmini"):
        self.env, self.task = env, task
        self.contract = extra_success_contract(task, backend)
        self.window = None

    def update(self, *, advanced=True):
        from benchmark.tasks.extra.contact import CONTACT_KINDS
        if self.task["episode_config"]["task_type"] in CONTACT_KINDS or self.task["episode_config"]["task_type"] in {"stand", "press_slide", "surface_inspection", "ram", "wipe_bottle", "wipe_whiteboard"}:
            return deepcopy(self.env.progress.result)
        # step() already sampled this control step. Match ExtraProgress's
        # no-advance rule before copying physical tensors back to the CPU.
        if not advanced or self.env.progress.last_sample == self.env.extra_step_counter:
            return dict(self.env.progress.result)
        return self.env.progress.update(self.env.read_task_state(), self.env.step_dt,
                                        sample_id=self.env.extra_step_counter, advanced=advanced)


def gel_surface_data(env, side):
    """Current surface/shear sample, independent of the native normal-force history."""
    name = f"gsmini_{side}_surface_contact"
    return env.scene[name if name in env.scene.keys() else f"gsmini_{side}_contact"].data


def make_extra_env_cfg(task, seed=0, *, device="cuda:0", cameras=True, physics_dt=None):
    """Reuse GSmini's robot/control/sensors; add rigid objects or articulations."""
    dt = task["episode_config"]["physics_dt"] if physics_dt is None else physics_dt
    ratio = task["episode_config"]["control_dt"]/dt if dt and math.isfinite(dt) else 0
    if ratio < 1 or not math.isclose(ratio, round(ratio), abs_tol=1e-8):
        raise ValueError("Extra physics_dt must be positive, finite and divide control_dt")
    import isaaclab.envs  # initialize envs before importing manager classes
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
    from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg, ImplicitActuatorCfg
    from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
    from isaaclab.utils import configclass
    from isaaclab.envs import mdp
    from benchmark.backends.gsmini.sensors import FrankaGsminiLiftEnvCfg, configure_contact_sensors
    from benchmark.catalog import object_asset
    from benchmark.backends.gsmini.environment import _configure_expert_env_cfg
    from benchmark.scene.cameras import configure_rgb_cameras
    from benchmark.scene.reset import robot_joint_defaults
    from benchmark.scene.visuals import configure_task_lighting

    cfg = FrankaGsminiLiftEnvCfg()
    _configure_expert_env_cfg(cfg, seed)
    cfg.sim.device = device
    cfg.sim.dt = dt
    cfg.decimation = round(task["episode_config"]["control_dt"] / cfg.sim.dt)
    cfg.sim.render_interval = cfg.decimation
    cfg.sim.enable_scene_query_support = True
    cfg.sim.physx.min_position_iteration_count = 16
    cfg.sim.physx.min_velocity_iteration_count = 16 if task["episode_config"]["task_type"] == "roll" else 4
    cfg.sim.physx.enable_external_forces_every_iteration = True
    if task["episode_config"]["task_type"] in {"roll", "gears", "hang", "slip", "stand", "press_slide"}:
        # Share the contact expert's PGS profile with collection and evaluation.
        cfg.sim.physx.solver_type = 0
    @configclass
    class RobotObservationsCfg(ObservationGroupCfg):
        joint_positions = ObservationTermCfg(func=mdp.joint_pos)
        joint_velocities = ObservationTermCfg(func=mdp.joint_vel)
        last_action = ObservationTermCfg(func=mdp.last_action)
        enable_corruption = False
        concatenate_terms = True

    cfg.observations.policy = RobotObservationsCfg()
    cfg.terminations.object_dropping = None
    cfg.terminations.success = None
    cfg.events.reset_robot_joints = None
    cfg.scene.robot.init_state.pos = tuple(task["robot_base_pos"])
    cfg.scene.robot.init_state.rot = tuple(task["robot_base_ori"])
    cfg.scene.robot.init_state.joint_pos = robot_joint_defaults(task)
    if task["episode_config"]["task_type"] == "slice":
        from benchmark.catalog import ASSETS
        cfg.scene.robot.spawn.usd_path = str(ASSETS / task["episode_config"]["robot_asset"])
    fixture = task["fixtures"]["living_room_table"]
    cfg.scene.table = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=fixture["initial_pose"], rot=fixture["initial_quat"]),
        spawn=sim_utils.UsdFileCfg(usd_path=str(object_asset(fixture["type"])), scale=tuple(fixture["scale"])))
    cfg.scene.object = None
    poses = sample_extra_reset(task, seed)

    class BistableActuator(IdealPDActuator):
        def compute(self, control_action, joint_pos, joint_vel):
            goal = task["goals"][0]
            self.computed_effort = switch_effort(joint_pos, joint_vel,
                **{k: goal[k] for k in ("well_rad", "potential_coefficient", "damping_nm_s")})
            self.applied_effort = self._clip_effort(self.computed_effort)
            control_action.joint_efforts = self.applied_effort
            control_action.joint_positions = control_action.joint_velocities = None
            return control_action

    for name, spec in task["objects"].items():
        pose = poses[name]
        spawn = sim_utils.UsdFileCfg(usd_path=str(object_asset(spec["type"])), scale=tuple(spec["scale"]),
                                     activate_contact_sensors=True)
        if task["episode_config"]["task_type"] == "ram" and name == "socket":
            from benchmark.tasks.extra.assembly import spawn_socket
            spawn.func = spawn_socket
        if task["episode_config"]["task_type"] in {"wipe_bottle", "wipe_whiteboard"} and name == "surface":
            from benchmark.tasks.extra.wiping import spawn_surface
            spawn.func = spawn_surface
        if task["episode_config"]["task_type"] == "slice":
            spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=64, solver_velocity_iteration_count=4,
                max_depenetration_velocity=0.3, linear_damping=0.1, angular_damping=0.1)
            if task["episode_config"].get("shared_scene") and name in task["episode_config"]["food_names"]:
                spawn.rigid_props.solver_velocity_iteration_count = 1
                if task["episode_config"]["original_task_type"] == "dice":
                    spawn.rigid_props.linear_damping = .3
                    spawn.rigid_props.angular_damping = 1.
        if spec["asset_kind"] == "articulation":
            actuators = {}
            if task["episode_config"]["task_type"] == "switch":
                actuators["bistable"] = IdealPDActuatorCfg(class_type=BistableActuator,
                    joint_names_expr=["toggle"], stiffness=0.0, damping=0.0,
                    effort_limit=1.0, effort_limit_sim=1.0)
            else:
                for joint, passive in spec.get("passive_actuators", {}).items():
                    actuators[joint] = ImplicitActuatorCfg(joint_names_expr=[joint],
                        stiffness=passive["stiffness"], damping=passive["damping"], effort_limit_sim=100.0)
            asset_cfg = ArticulationCfg(prim_path=f"{{ENV_REGEX_NS}}/{name}", spawn=spawn,
                init_state=ArticulationCfg.InitialStateCfg(pos=pose["position"], rot=pose["quaternion_wxyz"],
                                                          joint_pos=pose["joint_positions"]), actuators=actuators)
        else:
            asset_cfg = RigidObjectCfg(prim_path=f"{{ENV_REGEX_NS}}/{name}", spawn=spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=pose["position"], rot=pose["quaternion_wxyz"]))
        setattr(cfg.scene, name, asset_cfg)
    filters = [f"{{ENV_REGEX_NS}}/{name}/{body}"
               for name, spec in task["objects"].items() for body in spec["contact_bodies"]]
    configure_contact_sensors(cfg, filters)
    if task["episode_config"]["task_type"] == "slice":
        from isaaclab.sensors import ContactSensorCfg
        from benchmark.backends.gsmini.sensors import PhysicsStepContactSensor
        i = task["episode_config"]["cut_plane_index"]
        names = (task["episode_config"]["food_names"] if task["episode_config"].get("shared_scene") else
                 ["mechanism" if j == 4 else f"piece_{j}" for j in (i, i+1)])
        cfg.scene.knife_food_contact = ContactSensorCfg(
            class_type=PhysicsStepContactSensor,
            prim_path="{ENV_REGEX_NS}/knife/Piece", update_period=0.0,
            history_length=cfg.decimation,
            filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/{name}/Piece" for name in names])
    for side in ("left", "right"):
        contact = getattr(cfg.scene, f"gsmini_{side}_contact")
        if task["episode_config"]["task_type"] == "slice":
            # Normal histories remain at physics rate. Raw contact-buffer
            # aggregation synchronizes CUDA; surface/shear is consumed at 20 Hz.
            setattr(cfg.scene, f"gsmini_{side}_surface_contact", ContactSensorCfg(
                prim_path=contact.prim_path, update_period=task["episode_config"]["control_dt"],
                history_length=0, filter_prim_paths_expr=list(contact.filter_prim_paths_expr),
                track_contact_points=True, track_friction_forces=True,
                max_contact_data_count_per_prim=64))
            if task["episode_config"].get("shared_scene"):
                # Reuse GSmini's existing float64 periodic clock for long
                # all-seam episodes; keep Extra8/9's accepted path unchanged.
                getattr(cfg.scene, f"gsmini_{side}_surface_contact").class_type = PhysicsStepContactSensor
        else:
            contact.track_contact_points = True
            contact.track_friction_forces = True
            contact.max_contact_data_count_per_prim = 64
    from benchmark.tasks.extra.contact_runtime import CONTACT_KINDS, configure_contact_sensors as configure_task_contacts
    native_contacts = task["episode_config"]["task_type"] in CONTACT_KINDS or task["episode_config"]["task_type"] in {"stand", "press_slide", "surface_inspection", "ram", "wipe_bottle", "wipe_whiteboard"}
    if native_contacts:
        # Temporal DLAA retains the previous seeded pose across GPU resets.
        # FXAA refreshes the four camera streams without stepping physics.
        cfg.sim.render.antialiasing_mode = "FXAA"
        configure_task_contacts(cfg, task)
        if task["episode_config"]["task_type"] in {"wipe_bottle", "wipe_whiteboard"}:
            from benchmark.tasks.extra.wiping import configure_contacts
            configure_contacts(cfg)
        if task["episode_config"]["task_type"] == "ram":
            from benchmark.tasks.extra.assembly import configure_contacts
            configure_contacts(cfg)
        if task["episode_config"]["task_type"] == "surface_inspection":
            from benchmark.tasks.extra.inspection import configure_contacts
            configure_contacts(cfg)
        if task["episode_config"]["task_type"] == "press_slide":
            from benchmark.tasks.extra.latch import configure_contacts
            configure_contacts(cfg)
        if task["episode_config"]["task_type"] == "stand":
            cfg.scene.support_contact.filter_prim_paths_expr.append("{ENV_REGEX_NS}/cradle/Fixture")
    if task["episode_config"]["task_type"] in {"pivot", "cap"}:
        from isaaclab.sensors import ContactSensorCfg
        body = task["objects"]["mechanism"]["contact_bodies"][0]
        targets = (["support/Support"]
                   if task["episode_config"]["task_type"] == "pivot" else ["holder/Left", "holder/Right"])
        cfg.scene.support_contact = ContactSensorCfg(prim_path=f"{{ENV_REGEX_NS}}/mechanism/{body}",
            update_period=0.0, history_length=cfg.decimation,
            filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/{target}" for target in targets])
    configure_rgb_cameras(cfg.scene, suite="tac_extra")
    if cameras and task["episode_config"]["task_type"] == "slice":
        from benchmark.backends.gsmini.sensors import configure_marker_motion, tactile_marker_motion, TactileObservationsCfg
        configure_marker_motion(cfg, target_path=cfg.scene.knife.prim_path + "/Piece",
                                target_offset=tuple(task["reset_contract"]["tool_grip_local_m"]),
                                pose_motion=True)

        @configclass
        class CuttingTactileObservationsCfg(TactileObservationsCfg):
            left_marker_motion = ObservationTermCfg(func=tactile_marker_motion,
                params={"sensor_name": "gsmini_left"})
            right_marker_motion = ObservationTermCfg(func=tactile_marker_motion,
                params={"sensor_name": "gsmini_right"})

        cfg.observations.tactile = CuttingTactileObservationsCfg()
    if native_contacts:
        # The shared recording/conversion contract uses one RGB image shape.
        # Keep the front camera optics and all legacy task camera sizes intact.
        cfg.scene.wristview.width = cfg.scene.agentview.width
        cfg.scene.wristview.height = cfg.scene.agentview.height
    configure_task_lighting(cfg.scene, task)
    if not cameras:
        cfg.scene.agentview = cfg.scene.wristview = None
        cfg.scene.gsmini_left = cfg.scene.gsmini_right = None
        cfg.observations.vision = cfg.observations.tactile = None
    return cfg


class ExtraEnvironment:
    """Single-environment wrapper with explicit reset and shared joint actions.

    No autoreset or task-object controller is installed. The only nonrobot
    efforts are the documented springs, dampers and switch potential.
    """

    def __new__(cls, task_id, *args, **kwargs):
        from benchmark.catalog import get_task, is_extra_scene
        if cls is ExtraEnvironment and is_extra_scene(get_task("tac_extra", task_id)):
            from benchmark.tasks.extra.scenes import ExtraSceneEnvironment
            return ExtraSceneEnvironment(task_id, *args, **kwargs)
        return super().__new__(cls)

    def __init__(self, task_id, seed=0, *, device="cuda:0", cameras=True, physics_dt=None,
                 validate_reset=True, camera_shapes=None):
        import torch
        from isaaclab.envs import ManagerBasedEnv
        from benchmark.catalog import get_task

        self.task = get_task("tac_extra", task_id)
        cfg = make_extra_env_cfg(self.task, seed, device=device, cameras=cameras, physics_dt=physics_dt)
        if camera_shapes is not None:
            from benchmark.scene.cameras import configure_camera_shapes
            configure_camera_shapes(cfg.scene, camera_shapes)
        self.env = ManagerBasedEnv(cfg=cfg)
        self.scene, self.sim, self.cfg = self.env.scene, self.env.sim, self.env.cfg
        self.device, self.step_dt = self.env.device, self.env.step_dt
        self.num_envs, self.action_manager = 1, self.env.action_manager
        self.extra_step_counter = 0
        self.action = torch.zeros_like(self.action_manager.action)
        self.progress = ExtraProgress(self.task)
        self.cutting = None
        if self.task["episode_config"]["task_type"] == "slice":
            from benchmark.tasks.extra.cutting import CuttingRuntime
            if self.task["episode_config"].get("shared_scene"):
                from benchmark.tasks.extra.shared_cutting import SharedCuttingRuntime
                self.cutting = SharedCuttingRuntime(self)
            else:
                self.cutting = CuttingRuntime(self)
        self._resetting = False
        self.contact_runtime = None
        from benchmark.tasks.extra.contact_runtime import CONTACT_KINDS, ContactRuntime
        if self.task["episode_config"]["task_type"] in {"wipe_bottle", "wipe_whiteboard"}:
            from benchmark.tasks.extra.wiping import WipingRuntime
            self.contact_runtime = WipingRuntime(self)
        elif self.task["episode_config"]["task_type"] == "ram":
            from benchmark.tasks.extra.assembly import RamRuntime
            self.contact_runtime = RamRuntime(self)
        elif self.task["episode_config"]["task_type"] == "surface_inspection":
            from benchmark.tasks.extra.inspection import InspectionRuntime
            self.contact_runtime = InspectionRuntime(self)
        elif self.task["episode_config"]["task_type"] == "press_slide":
            from benchmark.tasks.extra.latch import LatchRuntime
            self.contact_runtime = LatchRuntime(self)
        elif self.task["episode_config"]["task_type"] == "stand":
            from benchmark.tasks.extra.placement import StandRuntime
            self.contact_runtime = StandRuntime(self)
        elif self.task["episode_config"]["task_type"] in CONTACT_KINDS:
            self.contact_runtime = ContactRuntime(self)
        self.validate_reset = validate_reset
        try:
            self._configure_cabinet_integrity()
            from benchmark.tasks.extra.visuals import apply_extra_materials
            apply_extra_materials(self)
            self.reset(seed=seed)
        except BaseException:
            from benchmark.backends.gsmini.execution import configure_shutdown
            configure_shutdown(self)
            self.env.close()
            raise

    def __getattr__(self, name):
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)

    @property
    def diagnostic_object(self):
        return self.scene["mechanism"]

    def reset(self, *, seed=0, options=None):
        if options:
            raise ValueError("Extra reset accepts seed only; physics parameters are fixed")
        self.reset_validation = []

        def accept(poses):
            from benchmark.tasks.extra.interaction import check_reachability, inspect_reset
            from benchmark.tasks.extra.visuals import native_view_report

            self._apply_reset_layout(seed, poses)
            check = inspect_reset(self) if self.cutting is None else {"passed": True, "errors": []}
            check["agentview"] = native_view_report(self)
            check["passed"] &= check["agentview"]["passed"]
            if not check["agentview"]["passed"]:
                check["errors"].append("AgentView framing or operation-surface visibility failed")
            if check["passed"]:
                check["reachability"] = check_reachability(self)
                check["passed"] = check["reachability"]["passed"]
            self.reset_validation.append(check)
            if not check["passed"]:
                failures = [r["waypoint"] for r in check.get("reachability", {}).get("waypoints", [])
                            if not r["reachable"]]
                print(f"[EXTRA RESET REJECT] task={self.task['task_id']} seed={seed} "
                      f"errors={check['errors']} unreachable={failures}",flush=True)
            return check["passed"]

        if self.validate_reset:
            try:
                sample_extra_reset(self.task, seed, reachable=accept,
                                   max_attempts=self.task["reset_contract"]["max_attempts"])
            except RuntimeError as error:
                last = self.reset_validation[-1] if self.reset_validation else {}
                failures = [r["waypoint"] for r in last.get("reachability", {}).get("waypoints", [])
                            if not r["reachable"]]
                raise RuntimeError(f"Extra task {self.task['task_id']} seed {seed}: {error}; "
                                   f"reset errors={last.get('errors', [])}; unreachable waypoints={failures}") from error
        else:
            self._apply_reset_layout(seed, sample_extra_reset(self.task, seed))
        return self.env.observation_manager.compute(), {"suite": "tac_extra", "task_id": self.task["task_id"],
            "seed": seed, "reset": deepcopy(self.reset_poses),
            "reset_attempts": sum("agentview" in row for row in self.reset_validation) or 1,
            "agentview_validated": bool(self.reset_validation and self.reset_validation[-1].get("agentview", {}).get("passed", False)),
            "reset_validated": bool(self.reset_validation and self.reset_validation[-1]["passed"])}

    def _apply_reset_layout(self, seed, poses):
        import torch
        from benchmark.scene.reset import robot_joint_defaults
        from benchmark.backends.gsmini.environment import _hold_current_arm_action

        self._resetting = True
        self.env.reset(seed=seed)
        if self.task["episode_config"]["task_type"] == "roll":
            from benchmark.tasks.extra.contact_assets import supported_roll_reset_poses
            poses = supported_roll_reset_poses(self, poses)
        if self.task["episode_config"]["task_type"] == "surface_inspection":
            from benchmark.tasks.extra.inspection import supported_reset_poses
            poses = supported_reset_poses(self, poses)
        if self.task["episode_config"]["task_type"] == "press_slide":
            from benchmark.tasks.extra.latch import supported_reset_poses
            poses = supported_reset_poses(self, poses)
        if self.task["episode_config"]["task_type"] == "stand":
            from benchmark.tasks.extra.placement import supported_reset_poses
            poses = supported_reset_poses(self, poses)
        robot = self.scene["robot"]
        joints = robot.data.default_joint_pos.clone()
        for name, value in robot_joint_defaults(self.task).items():
            joints[:, robot.joint_names.index(name)] = value
        robot.write_joint_state_to_sim(joints, torch.zeros_like(joints))
        robot.set_joint_position_target(joints)
        for name, spec in self.task["objects"].items():
            asset = self.scene[name]
            pose = poses[name]
            root = torch.tensor([pose["position"]+pose["quaternion_wxyz"]], device=self.device)
            root[:, :3] += self.scene.env_origins
            asset.write_root_pose_to_sim(root)
            asset.write_root_velocity_to_sim(torch.zeros((1, 6), device=self.device))
            if spec["asset_kind"] == "articulation":
                q = torch.zeros_like(asset.data.default_joint_pos)
                for joint, value in pose["joint_positions"].items():
                    q[:, asset.joint_names.index(joint)] = value
                asset.write_joint_state_to_sim(q, torch.zeros_like(q))
                target = torch.zeros_like(q)
                for joint, passive in spec.get("passive_actuators", {}).items():
                    target[:, asset.joint_names.index(joint)] = passive["target"]
                asset.set_joint_position_target(target)
        self.scene.write_data_to_sim()
        if self.contact_runtime is not None:
            from benchmark.tasks.extra.contact_runtime import reset_fixed_mounts
            reset_fixed_mounts(self, poses)
        if self.cutting is not None:
            self.cutting.reset()
        self.sim.forward()
        self.scene.update(0.0)
        self.scene.reset()
        self.action.zero_()
        _hold_current_arm_action(self, self.action)
        self.action[:, -1] = 1.0
        if self.cutting is not None:
            self.cutting.initialize_grasp()
            poses["knife"] = {"position": self.scene["knife"].data.root_pos_w[0].cpu().tolist(),
                              "quaternion_wxyz": self.scene["knife"].data.root_quat_w[0].cpu().tolist(),
                              "joint_positions": {}}
        for _ in range(round(1.0/self.step_dt)):
            self.env.step(self.action)
        if self.cutting is not None:
            extra_settle_steps = 0
            if self.task["episode_config"].get("shared_scene"):
                # Imported compliant foods can still roll after the common
                # one-second warmup. Hold the same physical grasp and wait;
                # never reset a moving food's pose or velocity to make it still.
                for _ in range(round(4.0/self.step_dt)):
                    speed = max(float(piece.data.root_lin_vel_w[0].norm()) for piece in self.cutting.pieces)
                    if speed <= self.task["goals"][0]["max_piece_speed_m_s"]:
                        break
                    self.env.step(self.action)
                    extra_settle_steps += 1
                self.shared_food_settle_steps = extra_settle_steps
            self.cutting.validate_reset()
            self.reset_validation.append({"passed": True, "kind": "cutting_intact_settled_grasp",
                                          **({"extra_food_settle_steps":extra_settle_steps}
                                             if self.task["episode_config"].get("shared_scene") else {}),
                                          "resting": self.cutting.metrics()})
        self.initial_state = self._snapshot()
        self.seed_value = seed
        self.reset_poses = poses
        self.progress = ExtraProgress(self.task)
        self.extra_step_counter = 0
        if self.contact_runtime is not None:
            self.contact_runtime.reset()
        self._resetting = False

    def _configure_cabinet_integrity(self):
        """Cache authored attachment frames and mesh offsets, never state targets."""
        self._cabinet_frames = None
        if self.task["episode_config"]["task_type"] not in {"drawer", "door"}:
            return
        import numpy as np
        from pxr import Usd, UsdGeom, UsdPhysics
        from scipy.spatial.transform import Rotation
        from benchmark.catalog import object_asset

        asset = self.scene["mechanism"]
        stage = Usd.Stage.Open(str(object_asset(self.task["objects"]["mechanism"]["type"])))
        self._cabinet_frames = []
        self._cabinet_geometry_gap_m = 0.0
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"])
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            parents, children = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if not parents or not children:
                continue  # The world mount is checked against its requested reset pose.
            def rotation(value):
                return Rotation.from_quat([*value.GetImaginary(), value.GetReal()])
            r0, r1 = rotation(joint.GetLocalRot0Attr().Get()), rotation(joint.GetLocalRot1Attr().Get())
            linear = prim.IsA(UsdPhysics.PrismaticJoint)
            fixed = prim.IsA(UsdPhysics.FixedJoint)
            axis = np.eye(3)["XYZ".index(prim.GetAttribute("physics:axis").Get())] if not fixed else None
            self._cabinet_frames.append({
                "parent": asset.body_names.index(parents[0].name),
                "child": asset.body_names.index(children[0].name),
                "p0": np.array(joint.GetLocalPos0Attr().Get()),
                "p1": np.array(joint.GetLocalPos1Attr().Get()), "r0": r0, "r1": r1,
                "axis": axis, "fixed": fixed,
                "dof": asset.joint_names.index(prim.GetName()) if not fixed else None,
                "linear": linear,
            })
        # The hinge point must lie on both the visible panel and its collider.
        # A valid joint between empty body frames cannot certify this condition.
        root = stage.GetDefaultPrim().GetPath()
        for side in ("left", "right"):
            joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(root.AppendPath(f"sektion/door_{side}_joint")))
            body = stage.GetPrimAtPath(joint.GetBody1Rel().GetTargets()[0])
            pivot = np.array(joint.GetLocalPos1Attr().Get())
            for group in ("visuals", "collisions"):
                box = bounds.ComputeRelativeBound(body.GetChild(group), body).ComputeAlignedRange()
                low, high = np.array(box.GetMin()), np.array(box.GetMax())
                gap = float(np.linalg.norm(np.maximum(np.maximum(low-pivot, pivot-high), 0)))
                self._cabinet_geometry_gap_m = max(self._cabinet_geometry_gap_m, gap)

    def _cabinet_integrity_state(self):
        """Measure attachment residuals from native live link transforms."""
        import numpy as np
        from scipy.spatial.transform import Rotation

        asset = self.scene["mechanism"]
        positions = asset.data.body_pos_w[0].cpu().numpy()
        quaternions = asset.data.body_quat_w[0].cpu().numpy()
        if not np.isfinite(positions).all() or not np.isfinite(quaternions).all():
            raise RuntimeError("Non-finite cabinet attachment transforms")
        rotations = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]])
        position_error = angle_error = 0.0
        for frame in self._cabinet_frames:
            a, b = frame["parent"], frame["child"]
            r0, r1 = rotations[a]*frame["r0"], rotations[b]*frame["r1"]
            p0 = positions[a]+rotations[a].apply(frame["p0"])
            p1 = positions[b]+rotations[b].apply(frame["p1"])
            value = float(asset.data.joint_pos[0, frame["dof"]]) if frame["dof"] is not None else 0.0
            if not math.isfinite(value):
                raise RuntimeError("Non-finite cabinet joint position")
            if frame["linear"]:
                p0 += r0.apply(frame["axis"])*value
            position_error = max(position_error, float(np.linalg.norm(p0-p1)))
            if frame["fixed"] or frame["linear"]:
                angle_error = max(angle_error, float((r0.inv()*r1).magnitude()))
            else:
                axis0, axis1 = r0.apply(frame["axis"]), r1.apply(frame["axis"])
                angle_error = max(angle_error, math.acos(float(np.clip(np.dot(axis0, axis1), -1, 1))))
                expected = Rotation.from_rotvec(frame["axis"]*value)
                angle_error = max(angle_error, float((expected.inv()*r0.inv()*r1).magnitude()),
                                  math.acos(float(np.clip(abs(axis0[2]), 0, 1))))
        root = asset.data.root_pos_w[0].cpu().numpy()
        expected = np.array(self.reset_poses["mechanism"]["position"])+self.scene.env_origins[0].cpu().numpy()
        mount_error = float(np.linalg.norm(root-expected))
        q = self.reset_poses["mechanism"]["quaternion_wxyz"]
        expected_rotation = Rotation.from_quat([*q[1:], q[0]])
        q = asset.data.root_quat_w[0].cpu().numpy()
        mount_angle = float((expected_rotation.inv()*Rotation.from_quat(q[[1, 2, 3, 0]])).magnitude())
        config = self.task["success_config"]
        intact = (position_error <= config["max_attachment_error_m"]
                  and angle_error <= config["max_attachment_angle_rad"]
                  and mount_error <= config["max_mount_error_m"]
                  and mount_angle <= config["max_attachment_angle_rad"]
                  and self._cabinet_geometry_gap_m <= config["max_hinge_geometry_gap_m"])
        return {"cabinet_intact": bool(intact), "cabinet_attachment_error_m": position_error,
                "cabinet_attachment_angle_rad": angle_error, "cabinet_mount_error_m": mount_error,
                "cabinet_mount_angle_rad": mount_angle,
                "cabinet_hinge_geometry_gap_m": self._cabinet_geometry_gap_m}

    def _snapshot(self):
        from scipy.spatial.transform import Rotation

        obj = self.scene["mechanism"]
        data = obj.data
        hand = self.scene["robot"]
        hand_index = hand.body_names.index("panda_hand")
        def rotation(q):
            w, x, y, z = q.cpu().tolist()
            return Rotation.from_quat([x, y, z, w])
        snapshot = {"position": data.root_pos_w[0].cpu().numpy().copy(),
                    "rotation": rotation(data.root_quat_w[0]),
                    "hand_rotation": rotation(hand.data.body_quat_w[0, hand_index])}
        if self.task["episode_config"].get("target_joint"):
            j = obj.joint_names.index(self.task["episode_config"]["target_joint"])
            snapshot["joint_position"] = float(data.joint_pos[0, j])
        return snapshot

    def read_task_state(self):
        if self.contact_runtime is not None and self.contact_runtime.initialized:
            return deepcopy(self.contact_runtime.last_sample)
        import numpy as np
        import torch
        from scipy.spatial.transform import Rotation

        def force(name):
            sensor = self.scene.sensors.get(name)
            if sensor is None:
                return 0.0
            values = sensor.data.force_matrix_w
            if values is None:
                raise RuntimeError(f"Missing filtered Extra contact data: {name}")
            return float(torch.linalg.vector_norm(values, dim=-1).sum())

        gel = sum(force(f"gsmini_{side}_contact") for side in ("left", "right"))
        case = sum(force(f"gsmini_{side}_case_contact") for side in ("left", "right"))
        state = {"gel_force_n": gel, "robot_force_n": gel+case}
        state.update({f"{side}_gel_force_n": force(f"gsmini_{side}_contact") for side in ("left","right")})
        state["gel_unfiltered_force_n"] = sum(float(torch.linalg.vector_norm(
            self.scene[f"gsmini_{side}_contact"].data.net_forces_w, dim=-1).sum()) for side in ("left", "right"))
        state["case_unfiltered_force_n"] = sum(float(torch.linalg.vector_norm(
            self.scene[f"gsmini_{side}_case_contact"].data.net_forces_w, dim=-1).sum()) for side in ("left", "right"))
        state["gel_contact_force_w_n"] = {side: self.scene[f"gsmini_{side}_contact"].data.force_matrix_w[0].cpu().tolist()
                                          for side in ("left", "right")}
        state["gel_friction_force_w_n"] = {side: gel_surface_data(self, side).friction_forces_w[0].cpu().tolist()
                                           for side in ("left", "right")}
        kind = self.task["episode_config"]["task_type"]
        obj = self.scene["mechanism"]
        now = self._snapshot()
        state.update(object_position_w_m=now["position"].tolist(),
                     object_quaternion_wxyz=obj.data.root_quat_w[0].cpu().tolist(),
                     tcp_position_w_m=self.scene["ee_frame"].data.target_pos_w[0,0].cpu().tolist(),
                     gripper_joint_positions=self.scene["robot"].data.joint_pos[0,-2:].cpu().tolist())
        if kind == "slice":
            state.update(self.cutting.metrics())
        if "joint_position" in now:
            j = obj.joint_names.index(self.task["episode_config"]["target_joint"])
            direction = self.task["episode_config"]["joint_direction"]
            state.update(joint_position=now["joint_position"], joint_speed=float(obj.data.joint_vel[0,j]),
                         joint_displacement=direction*(now["joint_position"]-self.initial_state["joint_position"]))
        if kind in {"drawer", "door"}:
            state.update(self._cabinet_integrity_state())
        if kind == "button":
            state["depth_m"] = -now["joint_position"]
        if kind in {"cap", "pivot"}:
            state["object_speed_m_s"] = float(torch.linalg.vector_norm(obj.data.root_lin_vel_w[0]))
        if kind == "cap":
            goal = self.task["goals"][0]
            holder = self.scene["holder"].data.root_pos_w[0].cpu().numpy()
            corners = np.array([[x,y,z] for x in (-0.017,0.017) for y in (-0.015,0.015)
                                for z in (goal["stem_bottom_local_m"],0.069)])
            bottom = float((now["rotation"].apply(corners)+now["position"])[:,2].min())
            state.update(withdrawal_m=float(now["position"][2]-self.initial_state["position"][2]),
                         clearance_m=float(bottom-holder[2]-goal["shoe_top_local_m"]),
                         holder_force_n=force("support_contact"))
        if kind == "pivot":
            relative = self.initial_state["rotation"].inv()*now["rotation"]
            current_grip = now["hand_rotation"].inv()*now["rotation"]
            reference = self.progress.grip_reference_xyzw
            slip_angle = (Rotation.from_quat(reference).inv()*current_grip).magnitude() if reference is not None else 0.0
            vector = relative.as_rotvec()
            state.update(pivot_rad=float(abs(vector[1])), off_axis_rad=float(np.linalg.norm(vector[[0,2]])),
                         relative_rotation_rad=float(slip_angle), grip_rotation_xyzw=current_grip.as_quat().tolist(),
                         table_force_n=force("support_contact"))
        if kind == "contour":
            side = max(range(2), key=lambda i: force(f"gsmini_{('left','right')[i]}_contact"))
            # The periodic surface sensor supplies shear only. Contact points
            # remain on the original physics-rate normal sensor.
            contacts = self.scene[f"gsmini_{('left','right')[side]}_contact"].data.contact_pos_w.reshape(-1, 3)
            valid = torch.isfinite(contacts).all(dim=-1)
            if not valid.any():
                return {**state, "path_bin": -1, "cross_track_m": 1.0}
            point = now["rotation"].inv().apply(contacts[valid].mean(dim=0).cpu().numpy()-now["position"])
            x = np.linspace(-0.08, 0.08, self.task["goals"][0]["bins"])
            y = 0.004*np.sin(2*np.pi*x/0.16)
            index = int(np.argmin(abs(x-point[0])))
            state.update(path_bin=index, cross_track_m=float(abs(abs(point[1]-y[index])-0.003)),
                         path_position_local_m=point.tolist())
        return state

    def step(self, action):
        import torch
        if action.shape != self.action.shape or not torch.isfinite(action).all():
            raise ValueError(f"Extra action must be finite with shape {tuple(self.action.shape)}")
        if action is not self.action:
            self.action.copy_(action)
        observations, info = self.env.step(self.action)
        if self.cutting is not None:
            self.cutting.update()
        self.extra_step_counter += 1
        if self.contact_runtime is not None:
            result = deepcopy(self.progress.result)
        else:
            result = self.progress.update(self.read_task_state(), self.step_dt, sample_id=self.extra_step_counter)
        timed_out = (not getattr(self, "policy_evaluation", False)
                     and self.extra_step_counter*self.step_dt >= self.task["episode_config"]["timeout_s"])
        if self.task["episode_config"]["task_type"] == "slice" and timed_out:
            result = {**result, "success": False, "reason": "cutting episode timed out; reset required"}
            self.progress.result = result
        return observations, {**info, "task_success": result, "terminated": result["success"],
                              "truncated": timed_out}

    def close(self):
        if self.contact_runtime is not None:
            self.contact_runtime.close()
        self.env.close()
