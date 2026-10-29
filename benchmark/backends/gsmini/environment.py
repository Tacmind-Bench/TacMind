"""Shared task scene and environment configuration for expert collectors."""

from __future__ import annotations

from copy import deepcopy
from contextlib import ExitStack
from functools import lru_cache

from benchmark.catalog import object_asset
from benchmark.backends.gsmini.geometry import _asset_geometry


class PlanningFailed(RuntimeError):
    """No valid plan was found; no trajectory has been executed."""


def _set_planner_seed(planner, seed):
    """Reseed cuRobo's episode-dependent samplers without replacing captured buffers."""
    import torch
    from curobo._src.util.sampling.sequencer_halton import HaltonSequencer

    # reset_seed() only restores the original seed in the installed cuRobo.
    # Other samplers (LM IK, PRM and optimizer noise) use fixed config seeds.
    for solver in (planner.ik_solver, planner.trajopt_solver):
        core = solver.core
        buffer = core.seed_manager.action_sample_generator
        buffer.sequencer = HaltonSequencer(ndims=buffer.ndims, seed=seed)
        if buffer._sample_buffer is not None:
            buffer._sample_buffer.copy_(torch.as_tensor(
                buffer.sequencer.random(buffer._store_buffer),
                device=buffer._sample_buffer.device, dtype=buffer._sample_buffer.dtype))
            buffer._int_gen.manual_seed(seed)
            buffer._generator_initial_state = buffer._int_gen.get_state().clone()
        core.config.random_seed = seed


class GSminiCollectionSession:
    """Keep a task's environment and warmed planner alive across episodes."""

    def __init__(self, suite: str, task_id: int):
        self.task_key = (suite, task_id)
        self.resources = ExitStack()
        self.env = None
        self.record_tactile = None
        self.planner = None
        self._planner_signature = None
        self._planner_capacity = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.env is not None:
            self.env.episode = None
        try:
            if self.planner is not None:
                self.planner.destroy()
        finally:
            self.resources.close()

    def bind(self, env):
        self.env = env
        self.record_tactile = getattr(env, "record_tactile", None)

    def reset(self, suite: str, task_id: int, seed: int):
        if (suite, task_id) != self.task_key:
            raise ValueError("Collection session cannot change tasks")
        env = self.env
        # Search wraps the tactile recorder and binds temporary candidate objects.
        # Episode resources have restored bindings before the next reset enters.
        env.episode = None
        env.grasp_reference = None
        env.grasp_center_local = None
        env.expert_success_monitor = None
        env.inserting = False
        env.record_tactile = self.record_tactile
        env.common_step_counter = env._sim_step_counter = 0
        if suite == "dynamic":
            from benchmark.tasks.dynamic.scene import prepare_dynamic_reset
            prepare_dynamic_reset(env, seed)
        env.reset(seed=seed)

    def acquire_planner(self, env, seed, world, *cache_worlds, **options):
        from benchmark.backends.gsmini.motion import _collision_cache_sizes, _create_motion_planner

        worlds = (world, *cache_worlds)
        versions = tuple(getattr(item, "_geometry_version", None) for item in worlds)
        signature = (str(env.device), options, versions)
        capacity = _collision_cache_sizes(*worlds)
        if self.planner is not None and (
                signature != self._planner_signature or None in versions
                or any(count > self._planner_capacity[kind] for kind, count in capacity.items())):
            # New geometry or larger caches can change pointers captured by cuRobo.
            previous, self.planner = self.planner, None
            previous.destroy()
        if self.planner is None:
            self.planner = _create_motion_planner(env, seed, world, *cache_worlds, **options)
            self._planner_signature, self._planner_capacity = signature, capacity
            self.planner.warmup(enable_graph=True, num_warmup_iterations=1)
        else:
            from curobo.types import ToolPoseCriteria
            from benchmark.backends.gsmini.workspace import update_collision_world

            self.planner.attachment_manager.detach()
            self.planner.update_tool_pose_criteria({"panda_hand": ToolPoseCriteria()})
            update_collision_world(self.planner, world)
            _set_planner_seed(self.planner, seed)
            self.planner.reset_seed()
        return self.planner


def environment_contract(env, task: dict, *, suite: str, backend: str) -> dict:
    """Record the task and observation/control settings used by this episode."""
    from benchmark.scene.cameras import camera_metadata
    from benchmark.contracts import GRIPPER_REPRESENTATION, gripper_action_metadata, joint_action_metadata

    result = {"suite": suite, "task_id": task["task_id"], "backend": backend,
              "task_definition": deepcopy(task),
              "control_dt": env.step_dt, "cameras": camera_metadata(env.cfg.scene),
              "gripper_action": GRIPPER_REPRESENTATION}
    if getattr(env, "action_manager", None) is not None:
        result["joint_action"] = joint_action_metadata(env)
        result["gripper_action"] = gripper_action_metadata(env)
    return result




@lru_cache(maxsize=32)
def rigid_body_suffix(asset_type: str) -> str:
    """Return the unique enabled rigid-body path relative to its USD root."""
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(object_asset(asset_type)))
    root = stage.GetDefaultPrim()
    bodies = [prim for prim in Usd.PrimRange(root)
              if prim.HasAPI(UsdPhysics.RigidBodyAPI)
              and UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is not False]
    if len(bodies) != 1:
        raise RuntimeError(f"Expected one rigid body in {asset_type}, found {len(bodies)}")
    return bodies[0].GetPath().pathString.removeprefix(root.GetPath().pathString)


def _configure_in_hand_joints(joint_positions: dict, task: dict) -> None:
    from benchmark.scene.reset import robot_joint_defaults

    joint_positions.pop("panda_finger_joint.*", None)
    joint_positions.update(robot_joint_defaults(task))


def _align_grasp_point_to_pads(env, point, midpoint):
    tcp = env.scene["ee_frame"].data.target_pos_w[:, 0]
    return point - (midpoint - tcp)


def _configure_task_scene_visuals(env, *, task=None) -> None:
    """Apply exposure while preserving authored Dynamic backgrounds and lights."""
    from benchmark.scene.visuals import apply_task_exposure_settings, hide_environment_visuals, replace_existing_lights

    task = task if task is not None else getattr(env, "task", {})
    if not task.get("scene_layout"):
        replace_existing_lights(env.sim.stage)
        hide_environment_visuals(env.sim.stage)
    exposure_settings = apply_task_exposure_settings()
    print(f"[VISUALS] Task RTX exposure settings enabled: {exposure_settings}", flush=True)


def initialize_task_environment(resources, env, seed: int, configure_shutdown,
                                *, task=None) -> None:
    """Set up and reset collection/evaluation environments in the same order."""
    resources.callback(env.close)
    _configure_task_scene_visuals(env, task=task)
    configure_shutdown(env)
    env.reset(seed=seed)


def _configure_expert_env_cfg(cfg, seed: int) -> None:
    cfg.scene.num_envs = 1
    cfg.seed = seed
    cfg.episode_length_s = 90.0
    cfg.events.reset_object_position = None
    cfg.commands = None
    cfg.rewards = None
    cfg.curriculum = None


def _configure_dynamic_env_cfg(cfg, seed: int, task: dict, stage, paths, root_pose) -> None:
    """Bind an expert config to the authored conveyor scene across backends."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from pxr import Usd
    from benchmark.scene.visuals import configure_task_lighting, stage_surface_z
    from benchmark.scene.cameras import configure_rgb_cameras

    _configure_expert_env_cfg(cfg, seed)
    cfg.sim.create_stage_in_memory = False
    # The lift task's world-height drop threshold assumes its own table origin.
    # Dynamic collectors use their measured grasp and insertion checks instead.
    cfg.terminations.object_dropping = None
    cfg.observations.policy = None
    cfg.scene.table = None
    cfg.scene.plane = None
    # Reused USD robots need the same physics overrides as freshly spawned ones.
    # Keep these edits in the session layer, as with the conveyor setup.
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        sim_utils.modify_rigid_body_properties(
            paths["robot"], cfg.scene.robot.spawn.rigid_props, stage=stage)
        sim_utils.modify_articulation_root_properties(
            paths["robot"], cfg.scene.robot.spawn.articulation_props, stage=stage)
    cfg.scene.robot.spawn = None
    cfg.scene.robot.prim_path = paths["robot"]
    configure_rgb_cameras(cfg.scene, paths["robot"], suite="tac_dynamic")
    cfg.scene.robot.init_state.pos = paths["_robot_pose"][0]
    cfg.scene.robot.init_state.rot = paths["_robot_pose"][1]

    for name in ("object", "fixture", "platform", "source_holder"):
        if name == "source_holder" and not paths[name]:
            continue
        position, quaternion = root_pose(stage, paths[name])
        setattr(cfg.scene, name, RigidObjectCfg(
            prim_path=paths[name],
            init_state=RigidObjectCfg.InitialStateCfg(pos=position, rot=quaternion),
            spawn=None,
        ))
    conveyor_path = "/World/ConveyorBeltPhysics"
    if task["task_id"] in (4, 6) and stage.GetPrimAtPath(conveyor_path):
        position, quaternion = root_pose(stage, conveyor_path)
        cfg.scene.conveyor = RigidObjectCfg(
            prim_path=conveyor_path, spawn=None,
            init_state=RigidObjectCfg.InitialStateCfg(pos=position, rot=quaternion))
    configure_task_lighting(cfg.scene, task, surface_z=stage_surface_z(stage, paths["table"]))


def _remap_robot_paths(scene, names, robot_path: str) -> None:
    """Bind robot-relative sensors and target frames to an authored scene robot."""
    for name in names:
        entity = getattr(scene, name, None)
        if entity is None:
            continue
        if hasattr(entity, "prim_path"):
            entity.prim_path = entity.prim_path.replace("{ENV_REGEX_NS}/Robot", robot_path)
        for frame in getattr(entity, "target_frames", ()):
            frame.prim_path = frame.prim_path.replace("{ENV_REGEX_NS}/Robot", robot_path)


def _hold_current_arm_action(env, action) -> None:
    """Keep task-specific reset joints through non-motion simulation steps."""
    arm = env.action_manager.get_term("arm_action")
    joint_ids, _ = env.scene["robot"].find_joints(
        arm.cfg.joint_names, preserve_order=arm.cfg.preserve_order,
    )
    action[:, :arm.action_dim] = arm.position_actions(
        env.scene["robot"].data.joint_pos[:, joint_ids],
    )


def reset_tabletop_task(env, task: dict, *, suite: str, seed: int,
                        allow_in_hand: bool = False, joint_defaults=None):
    """Apply the shared tabletop reset and return a command holding measured arm joints."""
    import torch
    from benchmark.scene.reset import reset_event

    options = {"joint_defaults": joint_defaults} if joint_defaults is not None else {}
    if "_eval_nominal_task" in task:
        options["task_override"] = task
    reset_event(env, [0], suite, task["task_id"], seed=seed,
                object_aliases={task["obj_of_interest"][0]: "object"},
                allow_in_hand=allow_in_hand, **options)
    action = torch.zeros_like(env.action_manager.action)
    _hold_current_arm_action(env, action)
    return action


def _make_env_cfg(seed: int, task: dict):
    """Build the shared task scene from catalog assets and reset poses."""
    import isaaclab.sim as sim_utils
    import torch
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.utils.math import quat_from_euler_xyz
    from benchmark.scene.reset import sample_reset
    from benchmark.scene.visuals import visual_material_cfg
    from benchmark.backends.gsmini.sensors import FrankaGsminiLiftEnvCfg

    cfg = FrankaGsminiLiftEnvCfg()
    _configure_expert_env_cfg(cfg, seed)
    # Keep task RGB/tactile observations; remove the inherited cube-lift terms.
    cfg.observations.policy = None
    fixture = task["fixtures"]["living_room_table"]
    cfg.scene.table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=fixture["initial_pose"], rot=fixture["initial_quat"]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(object_asset(fixture["type"])), scale=tuple(fixture["scale"])),
    )
    cfg.scene.robot.init_state.pos = tuple(task["robot_base_pos"])
    cfg.scene.robot.init_state.rot = tuple(task["robot_base_ori"])
    object_name = task["obj_of_interest"][0]
    contract = task.get("reset_contract") or {}
    poses = sample_reset(task, seed, allow_in_hand=contract.get("mode") == "in_hand")

    def object_cfg(name):
        spec = task["objects"][name]
        pose = poses[name]
        quaternion = pose.get("quaternion_wxyz")
        if quaternion is None:
            quaternion = quat_from_euler_xyz(*torch.tensor(pose["euler_xyz_rad"]).unbind()).tolist()
        rigid_props = {field: spec[field] for field in (
            "linear_damping", "angular_damping") if field in spec}
        spawn = sim_utils.UsdFileCfg(
            usd_path=str(object_asset(spec["type"])), scale=tuple(spec["scale"]),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(**rigid_props) if rigid_props else None,
            visual_material=visual_material_cfg(task, spec))
        return RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=pose["position"], rot=quaternion),
            spawn=spawn,
        )

    cfg.scene.object = object_cfg(object_name)
    for name in task["objects"]:
        if name != object_name:
            setattr(cfg.scene, name, object_cfg(name))
    if contract.get("mode") == "in_hand":
        _configure_in_hand_joints(cfg.scene.robot.init_state.joint_pos, task)
    from benchmark.scene.visuals import configure_task_lighting

    configure_task_lighting(cfg.scene, task)
    search = contract.get("mode") in {"blind_permutation", "exchange_layout"}
    if search or contract.get("mode") == "ball_sorting":
        from benchmark.catalog import candidate_names as _candidate_names
        contact_names = _candidate_names(task)
    else:
        contact_names = (object_name,)
    filters = [f"{{ENV_REGEX_NS}}/{name}{rigid_body_suffix(task['objects'][name]['type'])}"
               for name in contact_names]
    from benchmark.backends.gsmini.sensors import configure_contact_sensors
    configure_contact_sensors(cfg, filters, marker_target_path=(
        cfg.scene.object.prim_path + rigid_body_suffix(task['objects'][object_name]['type'])))
    return cfg
