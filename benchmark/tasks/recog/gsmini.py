"""Collect tactile recognition episodes with Franka + GSmini."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from functools import partial
from pathlib import Path

from benchmark.tasks.success import TaskSuccessMonitor, verify_task_success
from benchmark.catalog import get_task, candidate_names as _candidate_names
from benchmark.expert.capabilities import REUSABLE_GSMINI_TASK_IDS, supported_task_ids
from benchmark.backends.gsmini.environment import GSminiCollectionSession, _align_grasp_point_to_pads, initialize_task_environment, _make_env_cfg, reset_tabletop_task
from benchmark.backends.gsmini import geometry
from benchmark.backends.gsmini.execution import settle_scene as _settle_scene
from benchmark.backends.gsmini.motion import _attach_payload_for_planning, _check_grasp_motion, _create_motion_planner, _gripper_opening, _make_pose_controller, _move, _move_object_pose, _open_gripper, _tcp_object_pose, _servo_hand_pose
from benchmark.backends.gsmini.recording import _finish_episode, _prepare_output_path, _prepare_recording_step, _record_tactile_observations, _start_episode

TASK_IDS = supported_task_ids("recog", "gsmini")


def _search_order(candidate_count: int, seed: int):
    """Use a separate RNG stream so reset permutations cannot reveal the target."""
    import random

    # Preserve the historical stream so reclassification cannot change search order.
    return random.Random(f"tac_blind/search/{seed}").sample(range(candidate_count), candidate_count)


def _comparison_search_route(order, centers, target_slot, start):
    """Visit every candidate, ending at the teacher target to retain its grasp."""
    from itertools import permutations
    import math

    others = [slot for slot in order if slot != target_slot]
    def distance(route):
        points = [start, *(centers[slot] for slot in route), centers[target_slot]]
        return sum(math.dist(a, b) for a, b in zip(points, points[1:]))
    route = min(permutations(others), key=distance)
    return [*route, target_slot]


@contextmanager
def _candidate_binding(env, name: str, target_name: str):
    """Bind existing grasp helpers to a candidate without writing physical state."""
    if name == target_name:
        yield
        return
    objects = env.scene.rigid_objects
    objects["object"], objects[name] = objects[name], objects["object"]
    try:
        yield
    finally:
        objects["object"], objects[name] = objects[name], objects["object"]


def _search_until_match(order, probe, release, *, inspect_all=False):
    """Keep rejected contacts in the trajectory before testing the next slot."""
    matched = None
    for index, slot in enumerate(order):
        if probe(slot):
            if not inspect_all:
                return slot
            if matched is not None:
                raise RuntimeError("Recog search recognized multiple targets")
            matched = slot
            # All candidates have now been inspected. Keep this valid grasp
            # instead of releasing the selected last candidate and probing it again.
            if index == len(order) - 1:
                return matched
        release()
    if matched is not None:
        return matched
    raise RuntimeError("Recog search exhausted candidates without recognizing the target")


def _search_by_score(order, probe, release):
    """Compare contact measurements from all candidates, without identity labels."""
    import math

    scores = {}
    for slot in order:
        value = float(probe(slot))
        if not math.isfinite(value):
            raise RuntimeError("Recog search received an invalid tactile measurement")
        scores[slot] = value
        release()
    if len(scores) < 2:
        raise RuntimeError("Recog comparison requires at least two contact measurements")
    return max(scores, key=scores.get)


def _verify_candidate_contact(forces):
    """A missed grasp is an execution failure, never a negative identity label."""
    import torch

    if not torch.isfinite(forces).all() or not (forces.norm(dim=-1) > 0.0).all():
        raise RuntimeError(f"Recog search lacks bilateral contact: {forces.tolist()}")


def _configure_search_recording(env, candidates, slots):
    """Record candidate-filtered forces and labels outside policy observations."""
    import torch

    env.search_slot, env.search_phase, env.search_match = -1, 0, -1
    original_record_tactile = getattr(env, "record_tactile", None)

    def candidate_forces(slot):
        if slot < 0:
            return torch.zeros((2, 3), device=env.device)
        contact_index = candidates.index(slots[slot])
        return torch.stack([
            env.scene[f"gsmini_{side}_contact"].data.force_matrix_w[
                0, :, contact_index, :].sum(dim=0)
            for side in ("left", "right")])

    def record_search(episode):
        _record_tactile_observations(env, episode, original_record_tactile)
        episode.add("search/slot", torch.tensor(env.search_slot))
        episode.add("search/phase", torch.tensor(env.search_phase))
        episode.add("search/post_contact_match", torch.tensor(env.search_match))
        episode.add("search/contact_force_w", candidate_forces(env.search_slot).detach())

    env.record_tactile = record_search
    return candidate_forces


def _release_search_candidate(env, app, planner, action, attempts, *, wait_open=False,
                              retreat_height=0.12):
    """Open and retreat before restoring the candidate binding and search label."""
    from benchmark.backends.gsmini.execution import hold

    env.search_phase = 2
    action[:, -1] = 1.0
    if wait_open:
        _open_gripper(env, app, action)
    else:
        hold(env, app, action, 0.5)
    tcp = env.scene["ee_frame"].data
    retreat = tcp.target_pos_w[:, 0].clone()
    retreat[:, 2] += retreat_height
    _move(env, app, planner, action, retreat, "search_retreat",
          quaternion=tcp.target_quat_w[0, 0].tolist(), allow_gripper_contact=True)
    attempts.close()
    env.search_match = -1


def _placement_pose(env, task: dict, goal: dict):
    """Place the object over the task region using USD bounds and authored offsets."""
    import torch
    from curobo.types import Pose

    target_name = goal["target"]
    object_data = env.scene["object"].data
    target_data = env.scene[target_name].data
    center = geometry._object_center_local(env, task, goal.get("ref_local_offset"))
    local_center = center.expand_as(object_data.root_pos_w)
    center_offset = Pose(
        torch.zeros_like(object_data.root_pos_w), object_data.root_quat_w,
    ).transform_points(local_center)
    position = target_data.root_pos_w.clone() - center_offset

    relationship = goal["relationship"]
    if relationship == "retrieved_from":
        station = target_data.root_pos_w[0]
        robot = env.scene["robot"].data.root_pos_w[0]
        away_from_station = station[:2] - robot[:2]
        away_from_station /= torch.linalg.vector_norm(away_from_station).clamp_min(1e-6)
        distance_range = goal["distance_range"]
        position[0, :2] = station[:2] - away_from_station * min(
            max(0.19, float(distance_range[0]) + 0.04), float(distance_range[1]) - 0.04
        ) - center_offset[0, :2]
        support = geometry._collision_height(env, "Table", highest=True)
    elif relationship == "on":
        support = geometry._collision_height(env, target_name, highest=True)
    elif relationship == "in":
        # Lower the held object into the container before release; settling is
        # still resolved by PhysX contact, not by moving it to the floor.
        support = geometry._collision_height(env, target_name, highest=False) + 0.01
    else:
        raise ValueError(f"Unsupported TacRecog relationship: {relationship}")

    position[:, 2] = support - geometry._object_bottom_offset(env, task) + 0.003
    return Pose(position, object_data.root_quat_w.clone())


def _grasp_point(env, task: dict, *, support_clearance: float = 0.0):
    """Pinch balls at the equator and caps slightly above the USD center."""
    from curobo.types import Pose

    object_name = task["obj_of_interest"][0]
    spec = task["objects"][object_name]
    object_data = env.scene["object"].data
    point = Pose(object_data.root_pos_w, object_data.root_quat_w).transform_points(
        geometry._object_center_local(env, task).expand_as(object_data.root_pos_w))
    if spec["type"].startswith("blind_cap_"):
        point[:, 2] += 0.008 * float(spec["scale"][2])
    elif spec["type"].startswith(("factory_nut_", "blind_service_token_", "blind_fuse_")):
        # Clear the support with the housing, accounting for the object's height.
        minimum, maximum, _ = geometry._asset_geometry(spec["type"])
        half_height = (maximum[2] - minimum[2]) * float(spec["scale"][2]) / 2
        point[:, 2] += max(0.0, support_clearance - half_height)
    return point


def _grasp_with_contact(env, app, action, task, *, grasp_height=0.0,
                        contact_forces=None, quaternion=None, min_closing_seconds=0.5):
    """Follow the live object center; stop closure at bilateral contact."""
    import math
    import torch
    from benchmark.backends.gsmini.workspace import grasp_poses

    controller = _make_pose_controller(env)
    closing_steps = 0
    contact_steps = 0
    closing = False
    previous_center = None
    velocity = torch.zeros_like(env.scene["object"].data.root_pos_w)
    required = math.ceil(0.5 / env.step_dt)
    minimum = (required if contact_forces is None else
               math.ceil(min_closing_seconds / env.step_dt))
    for _ in range(math.ceil(8.0 / env.step_dt)):
        center = _grasp_point(env, task)
        center[:, 2] += grasp_height
        from benchmark.backends.gsmini.sensors import pad_centers_world

        midpoint = torch.stack(pad_centers_world(env)).mean(dim=0)
        center = _align_grasp_point_to_pads(env, center, midpoint)
        tcp = env.scene["ee_frame"].data.target_pos_w[:, 0]
        if torch.linalg.vector_norm(tcp - center, dim=-1).max() <= 0.001:
            closing = True
        if closing and contact_steps == 0:
            action[:, -1] = (action[:, -1] - 2.0 / required).clamp(min=-1.0)
        if previous_center is not None:
            velocity = 0.5 * velocity + 0.5 * (center - previous_center) / env.step_dt
        previous_center = center.clone()
        target = center.clone()
        target[:, :2] += 0.1 * velocity[:, :2]
        target[:, 2] = torch.maximum(center[:, 2], tcp[:, 2] - 0.005)
        _servo_hand_pose(env, app, controller, action, grasp_poses(env, target, quaternion=quaternion))
        if closing:
            closing_steps += 1
            if contact_forces is not None:
                forces = contact_forces()
                if not torch.isfinite(forces).all():
                    _verify_candidate_contact(forces)
                contact_steps = contact_steps + 1 if (forces.norm(dim=-1) > 0.0).all() else 0
            if closing_steps >= minimum and (contact_forces is None or contact_steps >= 2):
                return
    if contact_forces is not None:
        _verify_candidate_contact(contact_forces())
    raise RuntimeError(
        "Could not establish stable contact during physical grasp: "
        f"center={center.tolist()}, tcp={tcp.tolist()}, closing_steps={closing_steps}"
    )


def run(app, seed: int, output: Path, task_id: int, *,
        session: GSminiCollectionSession | None = None):
    if task_id not in TASK_IDS:
        raise ValueError(f"Expert unavailable for TacRecog {task_id}; choose from {TASK_IDS}")
    task = get_task("tac_recog", task_id)
    from benchmark.catalog import is_tabletop_task, tabletop_variant
    if is_tabletop_task(task):
        from benchmark.expert.tabletop import run as collect_tabletop
        return collect_tabletop(app, seed, output, tabletop_variant(task), session=session)
    if session is not None and task_id not in REUSABLE_GSMINI_TASK_IDS["recog"]:
        raise ValueError("Collection session is unavailable for this Recog task")
    if session is not None and session.task_key != ("recog", task_id):
        raise ValueError("Collection session cannot change tasks")
    _prepare_output_path(output)
    import math
    from curobo.types import Pose
    from isaaclab.envs import ManagerBasedRLEnv
    from benchmark.backends.gsmini.execution import configure_shutdown, execute_trajectory, hold, plan_grasp, settle_object, verify_grasp
    from benchmark.backends.gsmini.workspace import read_collision_world, update_collision_world

    object_name = task["obj_of_interest"][0]
    goal = next(g for g in task["goals"] if g["ref_obj"] == object_name)
    target_name = goal["target"]
    other_names = tuple(name for name in task["objects"] if name != object_name)
    tilted_grasp = task_id in (2, 3, 5, 6)
    # Tilt 20 degrees to avoid the folded elbow limit at these low grasps.
    grasp_options = ({"grasp_quaternion": (0.0, math.cos(math.pi / 18), 0.0, -math.sin(math.pi / 18))}
                     if tilted_grasp else {})

    class RecordingEnv(ManagerBasedRLEnv):
        episode = None
        grasp_reference = None
        grasp_center_local = None
        expert_success_monitor = None

        def step(self, action):
            _prepare_recording_step(self, action)
            result = super().step(action)
            if self.grasp_reference is not None:
                _check_grasp_motion(self)
            if self.expert_success_monitor is not None:
                self.expert_success_monitor.update()
            return result

    with ExitStack() as resources:
        fresh = session is None or session.env is None
        env = RecordingEnv(cfg=_make_env_cfg(seed, task)) if fresh else session.env
        env.compact_trajectory = True
        env.motion_tracking_wait_s = 1.0
        # Fuse search transfers hand control to another planned move or to the
        # precise contact servo; avoid settling every airborne hop to contact accuracy.
        env.motion_tracking_tolerance_rad = 0.02 if task_id == 5 else 0.005
        env.motion_free_settle_s = 0.0
        env.motion_insertion_tracking_tolerance_rad = 0.005 if task_id in (0, 1, 3, 5) else 0.10
        if fresh:
            initialize_task_environment(session.resources if session is not None else resources,
                                       env, seed, configure_shutdown, task=task)
            if session is not None:
                session.bind(env)
        else:
            session.reset("recog", task_id, seed)
        action = reset_tabletop_task(env, task, suite="tac_recog", seed=seed)
        action[:, -1] = 1.0
        _settle_scene(env, app, action, task)

        obstacles = ("Table", *other_names)
        world = read_collision_world(env, obstacles)
        object_world = read_collision_world(env, (*obstacles, object_name))
        planner_options = ({"trajectory_knots": 8 if task_id == 0 else 6}
                           if task_id in (0, 5) else {})
        if session is not None:
            planner = session.acquire_planner(env, seed, world, object_world, **planner_options)
        else:
            planner = resources.enter_context(
                _create_motion_planner(env, seed, world, object_world, **planner_options))
            planner.warmup(enable_graph=True, num_warmup_iterations=1)

        staged = _start_episode(resources, env, output, "tac_recog", task_id, seed, {
            "search_policy": ("shortest all-candidate route ending at the privileged teacher target" if task_id == 5 else
                              "independent seeded candidate permutation without replacement"),
            "search_phases": {"0": "approach", "1": "contact", "2": "release",
                              "3": "accepted", "4": "transfer"},
            "search_compare_all": task_id in (0, 5),
            "expert_target_source": (
                "measured gripper width comparison" if task_id == 0 else
                "post-contact task metadata teacher; not tactile-only recognition"),
        }, task=task)

        candidates = _candidate_names(task)
        # Physical slot labels are independent of object identity. The fuse teacher
        # uses identity only to retain the target after all six contact probes.
        def slot_position(name):
            with _candidate_binding(env, name, object_name):
                probe_task = dict(task, obj_of_interest=[name])
                return tuple(round(value, 3) for value in
                             _grasp_point(env, probe_task)[0, :2].tolist())

        original_target = object_name
        slots = sorted(candidates, key=slot_position)
        order = _search_order(len(slots), seed)
        if task_id == 5:
            order = _comparison_search_route(
                order, [slot_position(name)[:2] for name in slots], slots.index(original_target),
                env.scene["ee_frame"].data.target_pos_w[0, 0, :2].tolist())
        candidate_forces = _configure_search_recording(env, candidates, slots)
        attempts = ExitStack()
        resources.callback(attempts.close)
        grasp = None

        def probe(slot):
            nonlocal grasp
            name = slots[slot]
            attempts.enter_context(_candidate_binding(env, name, original_target))
            probe_task = dict(task, obj_of_interest=[name])
            env.search_slot, env.search_phase, env.search_match = slot, 0, -1
            env.grasp_center_local = None
            action[:, -1] = 1.0
            obstacle_names = tuple(n for n in task["objects"] if n != name)
            update_collision_world(planner, read_collision_world(env, ("Table", *obstacle_names)))
            center = _grasp_point(env, probe_task, support_clearance=(
                0.017 if task_id in (0, 2, 5, 6) else 0.0))
            grasp = plan_grasp(env, planner, grasp_position=center, lift_offset=-0.12,
                               approach_offset=-0.05 if task_id == 5 else -0.10, **grasp_options)
            execute_trajectory(env, app, grasp, "approach", action,
                               settle_seconds=0.0,
                               tracking_wait_s=1.0,
                               max_tracking_error=0.02 if task_id == 5 else 0.005)
            env.search_phase = 1
            if task["objects"][name]["type"].startswith("ball_class_"):
                _grasp_with_contact(env, app, action, probe_task,
                              contact_forces=lambda: candidate_forces(slot))
            else:
                if task_id == 5 and name != object_name:
                    # Pre-shape above the fuse; this opening still clears its body.
                    action[:, -1] = 0.0
                execute_trajectory(env, app, grasp, "grasp", action,
                                   settle_seconds=0.0,
                                   tracking_wait_s=1.0,
                                   # Fuse contact servo below still requires TCP
                                   # alignment and fresh bilateral target contact.
                                   max_tracking_error=0.005 if task_id == 5 else 0.0005)
                if task_id in (0, 2, 3, 5, 6):
                    height = (_grasp_point(env, probe_task, support_clearance=0.017)
                              - _grasp_point(env, probe_task))[0, 2].item()
                    _grasp_with_contact(env, app, action, probe_task, grasp_height=height,
                                        min_closing_seconds=(0.1 if task_id == 5
                                                             and name != object_name else 0.5),
                                        contact_forces=lambda: candidate_forces(slot),
                                        quaternion=(env.scene["ee_frame"].data.target_quat_w[:, 0].clone()
                                                    if tilted_grasp else None))
                else:
                    action[:, -1] = -1.0
                    hold(env, app, action, 0.5)
            for _ in range(2):
                hold(env, app, action, env.step_dt)
                forces = candidate_forces(slot)
                try:
                    _verify_candidate_contact(forces)
                except RuntimeError as error:
                    raise RuntimeError(
                        f"{error}; slot={slot}, opening={_gripper_opening(env).tolist()}, "
                        f"center={_grasp_point(env, probe_task).tolist()}, "
                        f"tcp={env.scene['ee_frame'].data.target_pos_w[:, 0].tolist()}"
                    ) from error
            if task_id == 0:
                width = float(_gripper_opening(env)[0])
                print(f"[SEARCH] slot={slot}, bilateral contact, width={width:.5f}m", flush=True)
                return width
            # Post-contact teacher supervision; this is not a tactile classifier.
            matched = name == object_name
            env.search_match = int(matched)
            env.search_phase = 3 if matched else 2
            hold(env, app, action, env.step_dt)
            print(f"[SEARCH] slot={slot}, bilateral contact, match={matched}", flush=True)
            return matched

        # Keep the small-cap search's release dwell before the next contact probe.
        release = partial(_release_search_candidate, env, app, planner, action, attempts,
                          wait_open=task_id != 7, retreat_height=0.05 if task_id == 5 else 0.12)

        # Relative questions require comparison, including all six threaded fuses.
        if task_id == 0:
            selected = _search_by_score(order, probe, release)
            probe(selected)
            env.search_match = 1
        else:
            selected = _search_until_match(order, probe, release, inspect_all=task_id == 5)
            if task_id == 5 and env.search_match != 1 and not probe(selected):
                raise RuntimeError("Recog search could not reidentify the selected candidate")
        verification_task = task
        object_name = slots[selected]
        object_type = task["objects"][object_name]["type"]
        other_names = tuple(name for name in task["objects"] if name != object_name)
        task = dict(task, obj_of_interest=[object_name])
        env.search_phase = 3
        if object_type.startswith("ball_class_"):
            _grasp_with_contact(env, app, action, task)
        object_data = env.scene["object"].data
        center_local = geometry._object_center_local(env, task)
        center_local_batch = center_local.expand_as(object_data.root_pos_w).clone()
        if object_type == "ball_class_golf":
            env.grasp_center_local = center_local_batch
        source_height = Pose(object_data.root_pos_w, object_data.root_quat_w).transform_points(
            center_local_batch)[:, 2].clone()
        env.grasp_reference = _tcp_object_pose(env)
        if object_type == "ball_class_golf":
            lift = env.scene["ee_frame"].data.target_pos_w[:, 0].clone()
            lift[:, 2] += 0.12
            _move(env, app, planner, action, lift, "lift", allow_gripper_contact=True)
        else:
            execute_trajectory(env, app, grasp, "lift", action,
                               settle_seconds=0.0,
                               tracking_wait_s=1.0,
                               max_tracking_error=0.005)
        verify_grasp(env, app, action, source_height,
                     min_height=0.04,
                     max_distance=0.04 if object_type == "ball_class_golf" else 0.10,
                     min_opening=0.01, object_center_local=center_local,
                     seconds=env.step_dt)

        env.search_phase = 4
        _attach_payload_for_planning(
            env, planner, other_names, read_collision_world, object_name,
        )
        destination = _placement_pose(env, task, goal)
        above = destination.position.clone()
        above[:, 2] += 0.10
        _move_object_pose(env, app, planner, action,
                          Pose(above, destination.quaternion), "transfer")

        final_obstacles = (other_names if goal["relationship"] == "retrieved_from"
                           else tuple(name for name in other_names if name != target_name))
        update_collision_world(planner, read_collision_world(env, ("Table", *final_obstacles)))
        if task_id == 3:
            # Keep the housing above the cup rim; the released cap falls inside.
            rim = geometry._collision_height(env, target_name, highest=True)
            tcp = destination.multiply(env.grasp_reference.inverse())
            destination.position[:, 2] += (rim + 0.03 - tcp.position[:, 2]).clamp_min(0.0)
        _move_object_pose(
            env, app, planner, action, destination, "insertion",
            settle_seconds=(0.0 if task_id in (0, 1, 3, 4, 5, 7) else
                            0.2 if task_id in (2, 6)
                            else None),
        )
        release_position = (Pose(object_data.root_pos_w, object_data.root_quat_w).transform_points(
            center_local_batch).clone() if object_type == "ball_class_golf" else
            object_data.root_pos_w.clone())
        action[:, -1] = 1.0
        env.grasp_reference = None
        env.grasp_center_local = None
        _open_gripper(env, app, action, require_fully_open=False)
        env.expert_success_monitor = TaskSuccessMonitor(
            env, verification_task, suite="tac_recog", backend="gsmini")
        update_collision_world(planner, read_collision_world(env, ("Table", *other_names, object_name)))
        planner.attachment_manager.detach()
        tcp = env.scene["ee_frame"].data
        retreat = tcp.target_pos_w[:, 0].clone()
        retreat[:, 2] += 0.10
        _move(env, app, planner, action, retreat, "retreat",
              quaternion=tcp.target_quat_w[0, 0].tolist(),
              allow_gripper_contact=True)
        settle_object(env, app, action,
                      max_seconds=(30.0 if goal["relationship"] == "in" else
                                   15.0 if (object_type == "ball_class_golf"
                                            or object_type.startswith("blind_fuse_")) else 4.0),
                      max_angular_speed=0.15 if object_type == "ball_class_golf" else 0.1,
                      max_displacement=0.20,
                      # An "on" placement may touch its support before opening.
                      require_drop=(goal["relationship"] != "in"
                                    and goal["relationship"] != "on"),
                      initial_position=release_position,
                      object_center_local=center_local if object_type == "ball_class_golf" else None)
        attempts.close()
        verify_task_success(env, app, action, verification_task,
                            suite="tac_recog", backend="gsmini", max_wait_seconds=30.0)
        _finish_episode(env, output, staged)
        print(f"[SUCCESS] Saved {output}", flush=True)
