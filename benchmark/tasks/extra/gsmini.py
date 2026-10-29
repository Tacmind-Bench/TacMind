"""GSmini Extra experts using physical contact, shared recording and success checks."""

from __future__ import annotations

from contextlib import ExitStack
import math
from pathlib import Path

from benchmark.expert.capabilities import supported_task_ids
from benchmark.backends.gsmini.recording import _finish_episode, _prepare_output_path, _prepare_recording_step, _record_tactile_observations, _start_episode
from benchmark.tasks.success import TaskSuccessMonitor, verify_task_success
from benchmark.tasks.extra.interaction import ContactMotion

TASK_IDS = supported_task_ids("extra", "gsmini")
PHASES = {"0": "lift", "1": "approach", "2": "press", "3": "withdraw", "4": "settle"}
MECHANISM_PHASES = {"0": "approach", "1": "contact", "2": "manipulate", "3": "release", "4": "settle"}


def _wait_for(motion, predicate, timeout, *, minimum=.1):
    """Advance real commands until two consecutive measured results agree."""
    confirmed = 0
    for index in range(math.ceil(timeout/motion.env.step_dt)):
        result = motion.step()
        confirmed = confirmed+1 if predicate(result) else 0
        if confirmed >= 2 and (index+1)*motion.env.step_dt >= minimum:
            return True
    return False


def _check_tool_grasp(env):
    import numpy as np

    runtime = env.cutting
    load_path = runtime.grasp_load_path()
    forces = load_path["gel_normal_force_n"]
    if not load_path["valid"]:
        raise RuntimeError(f"Cutting expert lost its gel load path: {load_path}")
    p, r = runtime.pose(runtime.knife)
    grip = p + r.apply(env.task["reset_contract"]["tool_grip_local_m"])
    tcp = env.scene["ee_frame"].data.target_pos_w[0, 0].cpu().numpy()
    gap = float(np.linalg.norm(grip-tcp))
    if (not np.isfinite(forces).all() or not math.isfinite(gap)
            or min(forces) < env.task["success_config"]["contact_threshold_n"] or gap > .015):
        raise RuntimeError(f"Cutting expert lost its physical knife grasp: forces={forces}, gap={gap}")


class MechanismMotion(ContactMotion):
    """Reuse validated contact and support feedback with the collector's app lifecycle."""

    def __init__(self, env, app):
        super().__init__(env)
        self.app = app

    def step(self):
        from benchmark.backends.gsmini.execution import SimulationClosed
        env, app = self.env, self.app
        while not env.sim.is_playing():
            if not app.is_running() or env.sim.is_stopped():
                raise SimulationClosed()
            env.sim.render()
        if not app.is_running() or env.sim.is_stopped():
            raise SimulationClosed()
        try:
            self._prepare_step()
            _, info = env.step(env.action)
        except Exception:
            if not app.is_running() or env.sim.is_stopped():
                raise SimulationClosed() from None
            raise
        if not app.is_running() or env.sim.is_stopped():
            raise SimulationClosed()
        if info["truncated"]:
            raise RuntimeError("Extra expert exceeded the task timeout")
        return info["task_success"]


class CuttingMotion(MechanismMotion):
    """Use the same bounded live Jacobian IK as tabletop contact motions."""

    def move(self, position, yaw, *, duration, quaternion=None):
        import torch
        from benchmark.backends.gsmini.workspace import grasp_poses
        from benchmark.backends.gsmini.motion import _hand_pose_joint_target

        env = self.env
        if env.task["episode_config"].get("shared_scene") and env.expert_phase in (0, 1) and duration >= 1.:
            return self.move_planned(position, yaw, duration=duration, quaternion=quaternion, payload=True)
        from benchmark.tasks.extra.shared_cutting import FAST_CONTACT_FOODS
        shared = env.task["episode_config"].get("shared_scene")
        food = env.task["episode_config"].get("food")
        fast_contact = shared and food in FAST_CONTACT_FOODS
        centered_stroke = shared and food == "baguette_1"
        max_step = .06 if shared and food == "baguette_2" and env.expert_phase == 2 else .04
        if shared and env.expert_phase == 3:
            max_step = .10 if food in {"cucumber_3", "cucumber_7", "carrot", "baguette_2", "banana",
                                       "baguette_1", "parsnip", "potato", "zucchini_1"} else .08 if fast_contact else .04
        point = torch.tensor([list(position)], dtype=torch.float32, device=env.device)
        quaternion = ([0, math.cos(yaw/2), math.sin(yaw/2), 0]
                      if quaternion is None else quaternion)
        controller = self.controller
        if centered_stroke and env.expert_phase == 2:
            from types import SimpleNamespace
            def centered_compute(position,quaternion,jacobian,joints):
                proposed=self.controller.compute(position,quaternion,jacobian,joints)
                bounds=env.scene["robot"].data.soft_joint_pos_limits[:,:7]
                projector=torch.eye(7,device=env.device)[None]-torch.linalg.pinv(jacobian)@jacobian
                centering=bounds.mean(dim=-1)-joints
                null=.01*(projector@centering[:,:,None]).squeeze(-1)
                room=torch.where(null>0,bounds[:,:,1]-proposed,proposed-bounds[:,:,0])
                fraction=torch.where(null.abs()>1e-12,room/null.abs().clamp_min(1e-12),torch.ones_like(null))
                scale=fraction.amin(dim=-1,keepdim=True).clamp(0.,1.)
                inside=((proposed>=bounds[:,:,0]) & (proposed<=bounds[:,:,1])).all(dim=-1,keepdim=True)
                return proposed+torch.where(inside,scale,0.)*null
            controller=SimpleNamespace(set_command=self.controller.set_command,compute=centered_compute)
        reached = 0
        for index in range(math.ceil(duration/env.step_dt)):
            goal = grasp_poses(env, point, quaternion=quaternion)
            target = _hand_pose_joint_target(env, controller, goal,
                                            max_joint_step=max_step,
                                            center_joints=not ((fast_contact and env.expert_phase in (2, 3))
                                                              or (centered_stroke and env.expert_phase == 2)))
            arm = env.action_manager.get_term("arm_action")
            env.action[:, :arm.action_dim] = arm.position_actions(target)
            self.step()
            if duration >= 1:
                from benchmark.tasks.extra.interaction import tcp_pose_error
                distance, angle = tcp_pose_error(env, point, quaternion)
                reached = reached+1 if distance < .003 and angle < .05 else 0
                if reached >= 2 and (index+1)*env.step_dt >= .25:
                    break


def _record_mechanism(env, episode):
    """Keep scalar simulator teacher state out of policy observations."""
    import torch

    _record_tactile_observations(env, episode)
    sample = env.read_task_state()
    state = {key: value for key, value in sample.items()
             if isinstance(value, (int, float))}
    if any(not math.isfinite(value) for value in state.values()):
        raise RuntimeError("Nonfinite Extra teacher state")
    episode.add("extra/phase", torch.tensor(env.expert_phase))
    for key, value in state.items():
        episode.add(f"extra/{key}", torch.tensor(value, dtype=torch.float64))
    for key in ("object_quaternion_wxyz", "relative_quaternion_wxyz"):
        value = sample.get(key)
        if value is not None:
            tensor = torch.as_tensor(value, dtype=torch.float64)
            if tensor.shape != (4,) or not torch.isfinite(tensor).all():
                raise RuntimeError(f"Invalid Extra teacher quaternion: {key}")
            episode.add(f"extra/{key}", tensor)
    runtime = getattr(env, "contact_runtime", None)
    if runtime is not None:
        fields = sorted(state)
        history = torch.zeros((env.cfg.decimation, len(fields)), dtype=torch.float64)
        valid = torch.zeros(env.cfg.decimation, dtype=torch.bool)
        samples = runtime.history[-env.cfg.decimation:]
        if samples:
            history[-len(samples):] = torch.tensor(
                [[sample[key] for key in fields] for sample in samples], dtype=torch.float64)
            valid[-len(samples):] = True
        if not torch.isfinite(history).all():
            raise RuntimeError("Nonfinite Extra native contact history")
        episode.add("extra/physics_state_history", history)
        episode.add("extra/physics_state_valid", valid)
        if getattr(runtime, "kind", None) in {"wipe_bottle", "wipe_whiteboard"}:
            distance = torch.as_tensor(env.progress.distance, dtype=torch.float64)
            episode.add("extra/wiping_cell_sliding_distance_m", distance)
            episode.add("extra/wiping_cleaned_mask", distance >= env.task["goals"][0]["cell_sliding_distance_m"])
            touched = torch.zeros((env.cfg.decimation, len(distance)), dtype=torch.bool)
            for index, sample in enumerate(samples, env.cfg.decimation-len(samples)):
                touched[index, sample["touched_cells"]] = True
            episode.add("extra/wiping_touched_cells_history", touched)
        for part in ("gel", "housing"):
            for component in ("normal", "tangent"):
                force = torch.tensor([
                    runtime.last_sample[f"{side}_{part}_{component}_force_w_n"]
                    for side in ("left", "right")], dtype=torch.float64)
                if not torch.isfinite(force).all():
                    raise RuntimeError("Nonfinite Extra native contact force")
                episode.add(f"extra/{part}_target_{component}_force_w", force)


def _record_cutting(env, episode):
    import torch

    _record_tactile_observations(env, episode)
    history = env.scene["knife_food_contact"].data.force_matrix_w_history[0, :, 0].flip(0)
    if not torch.isfinite(history).all():
        raise RuntimeError("Nonfinite knife-food contact history")
    episode.add("cutting/phase", torch.tensor(env.expert_phase))
    episode.add("cutting/cut_mask", torch.tensor(env.cutting.cut))
    episode.add("cutting/knife_pose_w", env.cutting.knife.data.root_state_w[0, :7].detach())
    if env.task["episode_config"].get("shared_scene"):
        episode.add("cutting/gates", torch.tensor(env.cutting.gates, dtype=torch.float64))
        episode.add("cutting/gate_valid", torch.tensor(env.cutting.gate_valid))
        episode.add("cutting/food_pose_w", torch.stack([
            piece.data.root_state_w[0,:7] for piece in env.cutting.pieces]).detach())
        episode.add("cutting/board_pose_w", env.scene["board"].data.root_state_w[0,:7].detach())
    episode.add("cutting/knife_food_normal_force_w_history", history.detach())
    episode.add("cutting/housing_force_fraction", torch.tensor(env.cutting.grasp_load_path()["housing_force_fraction"]))
    episode.add("cutting/gel_indentation_mm", torch.stack([
        env.scene[f"gsmini_{side}"].indentation_depth[0] for side in ("left", "right")]).detach())
    progress = getattr(env, "progress", None)
    # The shared monitor already measured this physical state after the last
    # control step. Reset/stale samples must still be read from the simulator.
    state = (progress.result["metrics"] if progress is not None and progress.last_sample is not None
             and progress.last_sample == env.extra_step_counter else env.read_task_state())
    for key, value in state.items():
        if isinstance(value, (int, float)):
            if not math.isfinite(value):
                raise RuntimeError(f"Nonfinite cutting teacher state: {key}")
            episode.add(f"cutting/state/{key}", torch.tensor(value, dtype=torch.float64))


def _make_recording_environment(task_id, seed, *, device="cuda:0", cameras=True):
    from benchmark.tasks.extra.environment import ExtraEnvironment

    class RecordingEnv(ExtraEnvironment):
        episode = None
        expert_phase = 0

        @property
        def common_step_counter(self):
            return self.extra_step_counter

        def step(self, action):
            _prepare_recording_step(self, action)
            result = super().step(action)
            if result[1]["truncated"]:
                raise RuntimeError("Extra expert exceeded the task timeout")
            if self.cutting is not None:
                _check_tool_grasp(self)
            return result

    return RecordingEnv(task_id, seed=seed, device=device, cameras=cameras)


def run(app, seed: int, output: Path, task_id: int, *, session=None):
    if task_id not in TASK_IDS:
        raise ValueError(f"Expert unavailable for Extra task {task_id}; GSmini supports {TASK_IDS}")
    from benchmark.catalog import get_task, is_extra_scene, is_tabletop_task, tabletop_variant
    task = get_task("tac_extra", task_id)
    if is_tabletop_task(task):
        from benchmark.expert.tabletop import run as collect_tabletop
        return collect_tabletop(app, seed, output, tabletop_variant(task), session=session)
    if session is not None:
        raise ValueError("Extra collection uses a fresh environment per episode")
    if is_extra_scene(task):
        from benchmark.tasks.extra.gsmini_scenes import run as run_scene
        return run_scene(app, seed, output, task_id)
    _prepare_output_path(output)
    from benchmark.backends.gsmini.execution import configure_shutdown

    with ExitStack() as resources:
        env = _make_recording_environment(task_id, seed)
        resources.callback(env.close)
        configure_shutdown(env)
        env.expert_success_monitor = TaskSuccessMonitor(env, env.task, suite="tac_extra", backend="gsmini")
        cutting = env.task["episode_config"]["task_type"] == "slice"
        recorder = _record_cutting if cutting else _record_mechanism
        env.record_tactile = lambda episode: recorder(env, episode)
        staged = _start_episode(resources, env, output, "tac_extra", task_id, seed, {
            "expert_target_source": "simulator geometry and physical-contact teacher; not tactile-only control",
            "expert_timing": ("adaptive-v2; confirmed contact/path arrival and completed-goal stop; required holds retained"
                              if env.task["episode_config"]["task_type"] in {"drawer", "contour"} else
                              "adaptive-v1; measured approach arrival and goal waits; contact servos and required holds retained"),
            **({"cutting_recording": {"phases": PHASES, "force_units": "N", "force_frame": "world",
                                      "descent_commands": ("Shared all-seam: <=4 mm in air, <=2 mm in contact; <=5 mm lead; completed score finish <=6 mm per tick and <=8 mm lead"
                                                           if env.task["episode_config"].get("shared_scene") else
                                                           "Cucumber: <=1 mm each control frame; banana: 2 mm per 0.15 s"),
                                      "history_order": "oldest to newest", "sample_alignment": "pre-action",
                                      "food_order": env.task["episode_config"].get("food_names", ["piece_3", "mechanism"]),
                                      **({"reset_extra_food_settle_steps": env.shared_food_settle_steps}
                                         if env.task["episode_config"].get("shared_scene") else {}),
                                      "gel_indentation_units": "mm; raw Taxim camera-depth estimate",
                                      "state": "scalar SI diagnostics under cutting/state/; separate from obs/"}} if cutting else
               {"extra_recording": {"phases": ({"0":"inspect_slot_0", "1":"inspect_slot_1",
                                                       "2":"inspect_slot_2", "3":"indicate_defect"}
                                                      if env.task["episode_config"]["task_type"] == "surface_inspection"
                                                      else MECHANISM_PHASES), "sample_alignment": "pre-action",
                                    "state": "scalar diagnostic truth under extra/; SI units"}}),
            "task_object_pose_writes_during_episode": 0,
            "task_definition": env.task,
            **({"extra_physics_recording": {
                "state_fields": sorted(key for key, value in env.read_task_state().items()
                                       if isinstance(value, (int, float))),
                "physics_dt": env.cfg.sim.dt, "history_order": "oldest to newest",
                "alignment": "pre-action; preceding native physics interval",
                "valid_mask": "extra/physics_state_valid; reset has no preceding interval",
                **({"wiping_cells": {
                    "field": "extra/wiping_touched_cells_history",
                    "cell_order": "benchmark.tasks.extra.wiping.surface_cells(task_definition)",
                    "meaning": "Boolean membership in the loaded native contact patch at each physics step"}}
                   if env.task["episode_config"]["task_type"] in {"wipe_bottle", "wipe_whiteboard"} else {}),
                "force_frame": "world", "force_units": "N", "force_sides": ["left", "right"],
                "force_scope": ("normal and tangent forces on the robot from all three inspection coupons"
                                if env.task["episode_config"]["task_type"] == "surface_inspection" else
                                "normal and tangent forces on the robot from the manipulated input body"),
                "policy_observations": "unchanged; all added state and force fields are diagnostics"}}
               if getattr(env, "contact_runtime", None) is not None else {}),
        }, task=env.task)
        if cutting:
            execute_cutting(env, CuttingMotion(env, app))
        else:
            execute_mechanism(env, MechanismMotion(env, app))
        result = verify_task_success(env, app, env.action, env.task, suite="tac_extra", backend="gsmini")
        if getattr(env, "contact_runtime", None) is not None:
            # Histories are pre-action. Execute one last hold to record the
            # successful native interval, which would otherwise be omitted.
            env.step(env.action)
            result = env.expert_success_monitor.update()
            if not result["success"]:
                raise RuntimeError(f"Extra completion hold lost native success: {result}")
        _finish_episode(env, output, staged)
        print(f"[SUCCESS] Saved {output}", flush=True)
        return result


def _lift_tool(env, motion, yaw):
    import numpy as np

    runtime = env.cutting
    tcp = env.scene["ee_frame"].data.target_pos_w[0, 0].cpu().numpy()
    lift=.04 if env.task["episode_config"].get("shared_scene") and runtime.config["food"] in {"banana","baguette_1"} else .07
    motion.move(tcp + [0, 0, lift], yaw, duration=4)
    for _ in range(3):
        if runtime.tool_lifted:
            return
        position, _ = runtime.pose(runtime.knife)
        remaining = max(.03, runtime.knife_reset_z+.04-float(position[2]))
        tcp = env.scene["ee_frame"].data.target_pos_w[0, 0].cpu().numpy()
        motion.move(tcp + np.array([0, 0, remaining]), yaw, duration=2)
    if not runtime.tool_lifted:
        raise RuntimeError(f"Cutting expert failed to lift the physical knife: {env.read_task_state()}")


def execute_cutting(env, motion):
    """Lift, align, press until measured release, withdraw and settle."""
    if env.task["episode_config"].get("shared_scene"):
        from benchmark.tasks.extra.shared_cutting import execute
        return execute(env, motion)
    import numpy as np

    runtime = env.cutting
    _, frame, _ = runtime.aim()
    normal = frame.apply([1, 0, 0])
    yaw = math.atan2(normal[1], normal[0]) + math.pi/2
    env.expert_phase = 0
    _lift_tool(env, motion, yaw)
    print(f"[CUT EXPERT] lifted={runtime.tool_lifted} forces={runtime.grasp_forces()}", flush=True)
    edge_local = runtime.edge_local[abs(runtime.edge_local[:, 0]-.1).argmin()]

    def move_edge(target, duration):
        kp, kr = runtime.pose(runtime.knife)
        edge = kp + kr.apply(edge_local)
        tcp = env.scene["ee_frame"].data.target_pos_w[0, 0].cpu().numpy()
        motion.move(tcp + np.asarray(target)-edge, yaw, duration=duration)

    env.expert_phase = 1
    aim, _, _ = runtime.aim()
    move_edge(aim + [0, 0, .05], 4)
    env.expert_phase = 2
    # Refresh the stiff cucumber's contact target every control frame. Holding
    # each 2 mm target for three frames creates velocity pulses at release.
    smooth = runtime.config.get("food") == "cucumber_2"
    descent = .001 if smooth else .002
    interval = env.step_dt if smooth else .15
    for step in range(math.ceil(.18/descent)):
        if any(runtime.cut):
            break
        kp, kr = runtime.pose(runtime.knife)
        edge = kp + kr.apply(edge_local)
        aim, _, _ = runtime.aim()
        target = aim.copy()
        target[2] = max(runtime.config["board_top_m"]+.001, edge[2]-descent)
        move_edge(target, interval)
        if step % 15 == 0:
            print(f"[CUT EXPERT] press={step} edge_z={edge[2]:.4f} cut={runtime.cut}", flush=True)
    if not any(runtime.cut):
        raise RuntimeError(f"Knife did not release the selected seam: {runtime.metrics()}")
    # Cohesion releases near the flesh midpoint. Finish the real downward
    # stroke so the blade wedges the remaining cross-section apart; a scored
    # plane flag alone is not a completed cut. Keep the cut line fixed once
    # the portions can move independently.
    cut_line, _, _ = runtime.aim()
    floor = runtime.config["board_top_m"]+.001
    for _ in range(math.ceil(.07/descent)):
        kp, kr = runtime.pose(runtime.knife)
        edge = kp + kr.apply(edge_local)
        if runtime.edge_world()[:, 2].min() <= floor+.001:
            break
        target = cut_line.copy()
        target[2] = max(floor, edge[2]-descent)
        move_edge(target, interval)
    else:
        raise RuntimeError(f"Knife could not complete its physical cutting stroke: {runtime.metrics()}")
    env.expert_phase = 3
    tcp = env.scene["ee_frame"].data.target_pos_w[0, 0].cpu().numpy()
    motion.move(tcp + [0, 0, .15], yaw, duration=3)
    env.expert_phase = 4
    _wait_for(motion, lambda result: result["success"], 1)


def execute_mechanism(env, motion):
    """Execute a reset-relative contact path with measured sliding/support feedback.

    Robot joint commands are the only outputs. Success remains owned by the
    environment's full process and stability checks, including release history.
    """
    from benchmark.tasks.extra.interaction import _rotation, contour_tcp_path, interaction_path, path_quaternions

    kind = env.task["episode_config"]["task_type"]
    from benchmark.tasks.extra.contact import CONTACT_KINDS
    if kind in {"wipe_bottle", "wipe_whiteboard"}:
        from benchmark.tasks.extra.wiping import execute_wiping
        return execute_wiping(env, motion)
    if kind == "ram":
        from benchmark.tasks.extra.assembly import execute_ram
        return execute_ram(env, motion)
    if kind == "surface_inspection":
        from benchmark.tasks.extra.inspection import execute_inspection
        return execute_inspection(env, motion)
    if kind == "press_slide":
        from benchmark.tasks.extra.latch import execute_latch
        return execute_latch(env, motion)
    if kind == "stand":
        from benchmark.tasks.extra.placement import execute_stand
        return execute_stand(env, motion)
    if kind in CONTACT_KINDS:
        from benchmark.tasks.extra.gsmini_contact import execute_contact
        return execute_contact(env, motion)
    if kind not in {"drawer", "door", "valve", "button", "switch", "cap", "contour", "pivot"}:
        raise ValueError(f"No mechanism expert for {kind}")
    path = interaction_path(env)
    quaternions = path_quaternions(env, path)
    env.expert_phase = 0
    if kind == "contour":
        targets, normals = contour_tcp_path(env, path)
        motion.normal_correction_limit_m = .003
        first, yaw = targets[0]
        normal = normals[0]
        motion.move(first-normal*0.011+[0,0,0.08], yaw, duration=.25, converge=True, rotation_tolerance=.08)
        motion.move(first-normal*0.001, yaw, duration=.25, converge=True, rotation_tolerance=.08)
        env.expert_phase = 2
        for index, ((p, yaw), normal) in enumerate(zip(targets, normals)):
            motion.surface_normal = normal
            def covered(result):
                state = result["metrics"]
                return (state["path_bin"] >= max(0, index-1)
                        and state["gel_force_n"] >= .1
                        and state["cross_track_m"] <= env.task["goals"][0]["cross_track_m"])
            motion.move(p, yaw, duration=.6, until=covered)
            if env.progress.result.get("success") and _wait_for(motion, lambda result: result["success"], .1):
                break
    else:
        for i, (p, yaw) in enumerate(path[:2]):
            # The pivot's very light pinch is sensitive to residual approach
            # motion. Retain its original settling budget and support history.
            motion.move(p, yaw, duration=2 if kind == "pivot" else .25, quaternion=quaternions[i],
                        converge=True, rotation_tolerance=None if kind == "pivot" else .08)
        env.expert_phase = 1
        env.action[:, -1] = -0.55 if kind == "pivot" else -1.0
        if kind == "pivot":
            motion.hold(0.8)
        else:
            _wait_for(motion, lambda result: result["metrics"]["robot_force_n"] >= .05
                      and float(env.scene["robot"].data.joint_vel[0, -2:].abs().max()) < .002,
                      .8, minimum=.2)
        if kind == "pivot":
            for _ in range(40):
                result = motion.step()["metrics"]
                if result["robot_force_n"] >= 0.6 and env.progress.contact_seen:
                    break
                env.action[:, -1] -= 0.005
            else:
                raise RuntimeError("Pivot expert could not establish a light tactile grasp")
        env.expert_phase = 2
        motion.pivoting = kind == "pivot"
        for i, (p, yaw) in enumerate(path[2:], 2):
            until = None
            if kind == "drawer":
                from benchmark.tasks.extra.interaction import tcp_pose_error
                until = lambda result: tcp_pose_error(env, p, quaternions[i])[0] < .001
            motion.move(p, yaw, duration=0.3 if kind == "pivot" else 0.2 if kind in {"door", "valve", "switch"}
                        else 0.6 if kind == "drawer" else 2,
                        quaternion=quaternions[i], until=until)
            if kind == "pivot" and env.read_task_state()["pivot_rad"] >= math.radians(89.8):
                break
        if kind == "drawer":
            layout = _rotation(env.reset_poses["mechanism"]["quaternion_wxyz"])
            for _ in range(3):
                error = 0.14-env.read_task_state()["joint_position"]
                if abs(error) < 0.0003:
                    break
                tcp = env.scene["ee_frame"].data.target_pos_w[0,0].cpu().numpy()
                motion.move(tcp+layout.apply([error,0,0]), path[-1][1], duration=1,
                            until=lambda result: abs(.14-result["metrics"]["joint_position"]) < .0003)
        if kind == "button":
            _wait_for(motion, lambda result: env.progress.pressed, .5)
        if kind == "cap":
            # Remove excessive grasp preload after extraction. The measured
            # jaw opening plus a 0.5 mm spring deflection requests ~1 N/jaw
            # with the existing 2000 N/m robot drive, without changing physics.
            q = float(env.scene["robot"].data.joint_pos[0,-2:].mean())
            env.action[:, -1] = max(-1.0, (q-0.0005)/0.02-1.0)
            _wait_for(motion, lambda result: result["success"], 2)
        if kind in {"button", "switch", "door"}:
            env.expert_phase = 3
            env.action[:, -1] = 1.0
            _wait_for(motion, lambda result: float(env.scene["robot"].data.joint_pos[0, -2:].min()) >= .035,
                      .5)
            p, yaw = path[-1]
            motion.move(p+[0,0,0.08], yaw, duration=.25, quaternion=quaternions[-1], converge=True,
                        rotation_tolerance=.08)
        env.expert_phase = 4
        if kind == "pivot":
            motion.hold(.8)
        else:
            _wait_for(motion, lambda result: result["success"], .8)
