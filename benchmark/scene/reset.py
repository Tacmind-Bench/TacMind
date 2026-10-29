"""Sample the authored tabletop reset layouts without loading Isaac Sim.

The uniform reset event handles independent rigid objects in a
ManagerBasedEnv. These suites additionally couple source assemblies and swap
blind candidates, so those task-specific relationships are sampled here.
"""

from __future__ import annotations

import math
import random
from copy import deepcopy
from typing import Callable, Sequence


# Downward GelSight Mini arm poses. The height-specific samples correspond to
# the authored fingertip heights of the special Tac tasks.
DEFAULT_ARM_JOINTS = (-0.0198828394, -0.1873406650, 0.0076694005,
                      -2.4034025985, 0.0049646816, 2.2453365042, 0.7948478983)
ARM_JOINTS_BY_FINGERTIP_HEIGHT = {
    0.07: (-0.0681724, 0.3508528, 0.0610103, -2.3893931, -0.0535264, 2.7391305, 0.8237857),
    0.12: (-0.0374469, 0.2222200, 0.0358500, -2.4011209, -0.0159408, 2.6231487, 0.7967650),
    0.15: (-0.0310978, 0.1514061, 0.0304961, -2.3980386, -0.0082383, 2.5493593, 0.7912806),
    0.22: (-0.0240020, 0.0085920, 0.0240063, -2.3618934, -0.0002959, 2.3704853, 0.7856088),
    0.24: (-0.0240289, -0.0257472, 0.0239926, -2.3442650, 0.0008423, 2.3185250, 0.7847812),
    0.28: (-0.0240827, -0.0944256, 0.0239652, -2.3090082, 0.0031187, 2.2146044, 0.7831260),
}


def robot_reset_contract(task: dict | None) -> dict:
    """Resolve the tabletop or dynamic contract used to initialize robot joints."""
    return ((task or {}).get("reset_contract") or
            (task or {}).get("episode_config", {}).get("in_hand_reset") or {})


def robot_joint_defaults(task: dict | None) -> dict[str, float]:
    contract = robot_reset_contract(task)
    height = contract.get("fingertip_height")
    arm = (ARM_JOINTS_BY_FINGERTIP_HEIGHT[float(height)] if height is not None
           else DEFAULT_ARM_JOINTS)
    if "arm_joint_positions" in contract:
        arm = contract["arm_joint_positions"]
        if len(arm) != 7 or not all(math.isfinite(q) for q in arm):
            raise ValueError("arm_joint_positions must contain seven finite positions")
    joints = {f"panda_joint{index}": value for index, value in enumerate(arm, 1)}
    width = float(contract.get("source_gripper_joint_pos",
                               contract.get("gripper_joint_pos", 0.04)))
    joints.update({f"panda_finger_joint{side}": width for side in (1, 2)})
    return joints


def _sample_range(rng: random.Random, bounds: Sequence[float], *,
                  context: str = "reset") -> float:
    if len(bounds) != 2:
        raise ValueError(f"Invalid {context} range: {bounds}")
    low, high = map(float, bounds)
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError(f"Invalid {context} range: {bounds}")
    return rng.uniform(low, high)


def _sample_pose(rng: random.Random, spec: dict, region: dict) -> dict:
    limits = region["pose_range"]
    pose = {"position": [_sample_range(rng, limits[axis]) for axis in ("x", "y", "z")]}
    if "initial_quat" in spec:
        pose["quaternion_wxyz"] = list(spec["initial_quat"])
    else:
        pose["euler_xyz_rad"] = [_sample_range(rng, limits[axis])
                                 for axis in ("roll", "pitch", "yaw")]
    return pose


def _sample_base(task: dict, rng: random.Random) -> dict[str, dict]:
    poses = {name: _sample_pose(rng, spec, task["regions"][spec["initial_region"]])
             for name, spec in task["objects"].items()}
    for group in task.get("randomization_groups", {}).values():
        limits = group["pose_range"]
        origin = [_sample_range(rng, limits[axis]) for axis in ("x", "y", "z")]
        angles = [_sample_range(rng, limits[axis]) for axis in ("roll", "pitch", "yaw")]
        for name, offset in group["members"].items():
            poses[name]["position"] = [origin[i] + offset[i] for i in range(3)]
            poses[name]["euler_xyz_rad"] = angles[:]
            poses[name].pop("quaternion_wxyz", None)

    for permutation in task.get("randomization_permutations", []):
        units = permutation["units"]
        slots = permutation["slot_centers"]
        if len(units) != len(slots):
            raise ValueError("Reset permutation has unequal units and slots")
        for unit, slot_index in zip(units, rng.sample(range(len(slots)), len(slots))):
            anchor = poses[unit[0]]["position"]
            slot = slots[slot_index]
            jitter = permutation["jitter"]
            half_widths = (jitter, jitter) if isinstance(jitter, (int, float)) else jitter
            delta = [slot[i] + rng.uniform(-half_widths[i], half_widths[i]) - anchor[i]
                     for i in (0, 1)]
            for name in unit:
                for axis in (0, 1):
                    poses[name]["position"][axis] += delta[axis]

    if "layout_xy_range" in task:
        delta = [_sample_range(rng, task["layout_xy_range"][axis]) for axis in ("x", "y")]
        for pose in poses.values():
            for axis in (0, 1):
                pose["position"][axis] += delta[axis]
    return poses


def _sample_precision(task: dict, rng: random.Random) -> dict[str, dict]:
    """Apply the three source task layouts whose JSON regions are spawn placeholders."""
    poses = _sample_base(task, rng)
    contract = task["reset_contract"]
    mode = contract["mode"]
    if mode in {"box_bottom", "object_search"}:
        from scipy.spatial.transform import Rotation

        box = poses[contract["box"]]
        box_rotation = Rotation.from_euler("xyz", box["euler_xyz_rad"])
        if mode == "box_bottom":
            part_rotation = box_rotation * Rotation.from_euler(
                "xyz", contract["rotation_euler_xyz_rad"])
            # The peg collider spans +/-9.91 mm across its local Y axis.
            bottom = min(part_rotation.apply((0, y, z))[2]
                         for y in (-0.00991, 0.00991) for z in (0.006, 0.106))
            offset = [*contract["initial_local_xy"],
                      contract["floor_top_local_z"] + contract["floor_clearance_m"] - bottom]
            part = contract["object"]
            poses[part] = {"position": (box_rotation.apply(offset) + box["position"]).tolist(),
                           "euler_xyz_rad": part_rotation.as_euler("xyz").tolist()}
        else:
            offset = [rng.choice((0.0, 0.05)) - contract["object_center_local_xy"][0],
                      -contract["object_center_local_xy"][1],
                      contract["height_above_box_root"]]
            poses[contract["object"]] = {
                "position": (box_rotation.apply(offset) + box["position"]).tolist(),
                "euler_xyz_rad": box_rotation.as_euler("xyz").tolist()}
    elif mode == "ball_sorting":
        base = poses[contract["base"]]["position"]
        golf_y, pingpong_y = ((0.05, -0.05) if rng.randrange(2) else (-0.05, 0.05))
        centers = {contract["golf"]: (0.0, golf_y),
                   contract["table_tennis"]: (0.0, pingpong_y),
                   contract["blank"]: (rng.choice((-0.10, 0.10)), 0.0)}
        # Measured root-to-centre offsets and support heights of the three USDs.
        geometry = {contract["golf"]: (0.0, 0.052, 0.0095),
                    contract["table_tennis"]: (0.0, 0.0, 0.0300),
                    contract["blank"]: (0.0, -0.0544, 0.0100)}
        for name, (center_x, center_y) in centers.items():
            root_x, root_y, height = geometry[name]
            poses[name] = {"position": [base[0] + center_x - root_x + rng.uniform(-0.012, 0.012),
                                         base[1] + center_y - root_y + rng.uniform(-0.012, 0.012),
                                         base[2] + height],
                           "euler_xyz_rad": [0.0, 0.0, 0.0]}
    return poses


def _sample_blind(task: dict, rng: random.Random) -> dict[str, dict]:
    contract = task["reset_contract"]
    if contract["mode"] == "exchange_layout":
        poses = deepcopy(contract["base_poses"])
        xy_only = {frozenset(group) for group in contract.get("exchange_xy_only_groups", [])}
        offsets = contract.get("exchange_center_xy_offsets", {})
        for group in contract["exchange_groups"]:
            original = [deepcopy(poses[name]) for name in group]
            slots = rng.sample(range(len(group)), len(group))
            keep_height = frozenset(group) in xy_only
            for name, index in zip(group, slots):
                if keep_height:
                    source_offset = offsets.get(group[index], (0.0, 0.0))
                    own_offset = offsets.get(name, (0.0, 0.0))
                    poses[name]["position"][:2] = [
                        original[index]["position"][axis] + source_offset[axis] - own_offset[axis]
                        for axis in (0, 1)
                    ]
                else:
                    poses[name] = original[index]
        delta = [_sample_range(rng, contract["shared_xy_range"][axis]) for axis in ("x", "y")]
        for pose in poses.values():
            for axis in (0, 1):
                pose["position"][axis] += delta[axis]
        return poses

    if contract["mode"] == "blind_permutation":
        poses = _sample_base(task, rng)
        station = poses[contract["station"]]
        for axis, half_width in enumerate(contract["station_xy_jitter"]):
            station["position"][axis] += rng.uniform(-half_width, half_width)
        roles = contract["roles"].values()
        slots = rng.sample(contract["cell_centers"], len(roles))
        for role, slot in zip(roles, slots):
            jitter = contract["jitter_half_xy"]
            local = [slot[axis] + rng.uniform(-jitter[axis], jitter[axis])
                     for axis in (0, 1)]
            if "support_cradle" in contract:
                cradle = contract["support_cradle"]
                # Seat the sphere above its nearest rail after XY jitter.
                edge_distance = min(cradle["half_xy"][axis] - abs(local[axis] - slot[axis])
                                    for axis in (0, 1))
                radius = role["collision_radius"]
                if not 0 <= edge_distance <= radius:
                    raise ValueError("Ball reset is outside its support cradle")
                height = (cradle["top_z"] + math.sqrt(radius**2 - edge_distance**2)
                          + cradle["clearance_m"])
            else:
                height = role["z"]
            local.append(height)
            center_offset = role.get("geometry_center_offset", (0.0, 0.0, 0.0))
            poses[role["object"]]["position"] = [station["position"][axis]
                                                   + local[axis] - center_offset[axis]
                                                   for axis in range(3)]
        return poses

    raise ValueError(f"Unknown blind reset mode: {contract['mode']}")


def sample_reset(task: dict, seed: int,
                 reachable: Callable[[dict[str, dict]], bool] | None = None,
                 max_attempts: int = 32, allow_in_hand: bool = False) -> dict[str, dict]:
    """Return reproducible environment-local poses without changing physics.

    In-hand tasks are rejected unless allow_in_hand is set; that option returns
    their placeholder layout only, so callers must apply the calibrated grasp
    before stepping or collecting. Conveyor control remains unsupported. If a
    reachability check is supplied, reject and resample the entire layout.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if task.get("reset_contract", {}).get("mode") == "extra":
        from benchmark.tasks.extra.environment import sample_extra_reset

        return sample_extra_reset(task, seed, reachable=reachable, max_attempts=max_attempts)
    if "_eval_nominal_task" in task:
        from benchmark.scene.perturbations import perturb_reset_poses
        nominal = task["_eval_nominal_task"]
        poses = sample_reset(nominal, seed, reachable=None, max_attempts=max_attempts,
                             allow_in_hand=allow_in_hand)
        if reachable is not None:
            raise ValueError("Perturbed policy resets do not use expert reachability resampling")
        return perturb_reset_poses(nominal, poses, task["_eval_object_scale"])
    rng = random.Random(seed)
    contract = task.get("reset_contract") or {}
    mode = contract.get("mode")
    if contract and mode not in {"tabletop", "exchange_layout", "blind_permutation",
                                "box_bottom", "object_search", "ball_sorting"} \
            and not (allow_in_hand and mode == "in_hand"):
        raise NotImplementedError(f"Reset mode {contract['mode']} needs robot control")
    if task.get("scene_layout"):
        raise NotImplementedError("Conveyor reset needs platform motion control")
    for _ in range(max_attempts if reachable is not None else 1):
        poses = (_sample_blind(task, rng) if mode in {"exchange_layout", "blind_permutation"}
                 else _sample_precision(task, rng) if mode else _sample_base(task, rng))
        if reachable is None or reachable(poses):
            return poses
    raise RuntimeError(f"No reachable reset layout after {max_attempts} attempts")


class CuRoboGraspCheck:
    """Check a candidate with cuRobo's collision-aware approach/grasp/lift plan.

    The caller supplies a GelSight-aware planner and two task-specific builders.
    The world builder must include every fixture and non-grasped object at its
    candidate pose, while excluding the object to grasp. The goal builder must
    return GoalToolPose in the robot base frame, using the configured TCP.
    Neither builder may change the simulation; rejected candidates are never
    written to PhysX. Construct the planner once and destroy it with the env.
    """

    def __init__(self, planner, world_for_candidate, goals_for_candidate,
                 approach_clearance_m: float = 0.10, lift_clearance_m: float = 0.20):
        for name, value in (("approach_clearance_m", approach_clearance_m),
                            ("lift_clearance_m", lift_clearance_m)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.planner = planner
        self.world_for_candidate = world_for_candidate
        self.goals_for_candidate = goals_for_candidate
        self.approach_clearance_m = approach_clearance_m
        self.lift_clearance_m = lift_clearance_m

    def __call__(self, env, task: dict, poses: dict[str, dict], env_id: int) -> bool:
        from curobo.types import JointState

        world = self.world_for_candidate(env, task, poses, env_id)
        goals = self.goals_for_candidate(env, task, poses, env_id)
        self.planner.update_world(world)
        robot = env.scene["robot"]
        joint_pos = robot.data.joint_pos[env_id:env_id + 1].clone()
        for name, value in robot_joint_defaults(task).items():
            if name not in robot.joint_names:
                raise ValueError(f"GelSight Mini reset joint missing: {name}")
            joint_pos[:, robot.joint_names.index(name)] = value
        current = JointState.from_position(
            joint_pos, joint_names=robot.joint_names,
        ).reorder(self.planner.joint_names)
        result = self.planner.plan_grasp(
            grasp_poses=goals, current_state=current,
            grasp_approach_offset=-self.approach_clearance_m,
            grasp_lift_offset=-self.lift_clearance_m,
            disable_collision_links=[],
        )
        return result.success is not None and bool(result.success.all().item())


def reset_event(env, env_ids, suite: str, task_id: int, seed: int | None = None,
                reachability: Callable[[object, dict, dict[str, dict], int], bool] | None = None,
                max_attempts: int = 32,
                object_aliases: dict[str, str] | None = None,
                allow_in_hand: bool = False,
                joint_defaults: Callable[[dict], dict[str, float]] = robot_joint_defaults,
                task_override: dict | None = None) -> None:
    """Isaac Lab reset event for the robot and task objects.

    Apply after the default robot reset and before subsequent physics settling.
    Robot joints and object poses are written through Isaac Lab's asset API.
    Supply a configured CuRoboGraspCheck to reject unreachable layouts before
    any object is written. The caller must settle and revalidate after reset.
    For in-hand tasks, allow_in_hand only permits the sampled placeholder; the
    caller must initialize the documented object-in-gripper pose before stepping.
    """
    import torch
    from isaaclab.utils.math import quat_from_euler_xyz

    from benchmark.catalog import get_task

    task = task_override if task_override is not None else get_task(suite, task_id)
    if task["task_id"] != task_id:
        raise ValueError("Reset override task ID differs from the requested task")
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if ids.ndim == 0:
        ids = ids.unsqueeze(0)
    if ids.numel() == 0:
        return
    plans = [sample_reset(
        task, seed + int(env_id) if seed is not None
        else int(torch.randint(0, 2**31 - 1, ()).item()),
        reachable=(lambda poses, env_id=env_id: reachability(env, task, poses, env_id))
        if reachability is not None else None,
        max_attempts=max_attempts,
        allow_in_hand=allow_in_hand,
    ) for env_id in ids.tolist()]
    robot = env.scene["robot"]
    joint_values = joint_defaults(task)
    if not set(joint_values) <= set(robot.joint_names):
        raise ValueError("Task reset requires joints missing from the configured robot")
    joint_pos = robot.data.default_joint_pos[ids].clone()
    for name, value in joint_values.items():
        joint_pos[:, robot.joint_names.index(name)] = value
    object_roots = []
    for name in task["objects"]:
        asset = env.scene[(object_aliases or {}).get(name, name)]
        object_poses = [plan[name] for plan in plans]
        positions = torch.tensor([pose["position"] for pose in object_poses], device=env.device)
        positions += env.scene.env_origins[ids]
        if "quaternion_wxyz" in object_poses[0]:
            quaternions = torch.tensor(
                [pose["quaternion_wxyz"] for pose in object_poses],
                device=env.device, dtype=positions.dtype,
            )
        else:
            angles = torch.tensor(
                [pose["euler_xyz_rad"] for pose in object_poses],
                device=env.device, dtype=positions.dtype,
            )
            quaternions = quat_from_euler_xyz(*angles.unbind(dim=-1))
        root_pose = torch.cat((positions, quaternions), dim=1)
        object_roots.append((asset, root_pose))
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=ids)
    robot.set_joint_position_target(joint_pos, env_ids=ids)
    for asset, root_pose in object_roots:
        asset.write_root_pose_to_sim(root_pose, env_ids=ids)
        asset.write_root_velocity_to_sim(torch.zeros((len(ids), 6), device=env.device),
                                         env_ids=ids)
