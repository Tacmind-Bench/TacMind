"""EmbodiedSWE scenes with TacMind's unmodified Franka + GSmini runtime.

Import-light catalog/config helpers; construct the environment after AppLauncher.
Upstream scene/grader code stays pinned and read-only in third_party/robobench.
"""

from __future__ import annotations

from copy import deepcopy
from importlib import import_module
import math
from types import SimpleNamespace

from benchmark.catalog import get_task
from benchmark.tasks.extra.scene_assets import upstream, relocate_config_assets


def physics_decimation(dt):
    ratio = 0.05 / dt if isinstance(dt, (int, float)) and math.isfinite(dt) and dt > 0 else 0
    if ratio < 1 or not math.isclose(ratio, round(ratio), abs_tol=1e-8):
        raise ValueError("Extra scene physics dt must be positive, finite and divide the 20 Hz control interval")
    return round(ratio)


def task_binding(task):
    """Reuse scene presets; carry only placement from the upstream robot."""
    upstream()
    source = task["source"]
    import_module(f"robobench.suites.{source['suite']}")
    from robobench.core import ENVS, SCENES
    name = f"{source['suite']}.{source['scene']}"
    if source["variant"]:
        name += "_" + source["variant"]
    if name + ".franka.joint" in ENVS.list():
        binding = ENVS.get(name + ".franka.joint")()
    else:
        binding = ENVS.get(name)()
        # These tasks have no single-arm upstream preset. Preserve the scene's
        # dimensions/mechanics; place the existing Panda south of the workspace.
        cfg = binding.scene_cfg or SCENES.get(source["scene"])().cfg
        height = float(cfg.surface_z)
        y = -0.65 if source["scene"] in {"push_shapes", "classify_objects"} else -0.45
        binding.scene_cfg = cfg
        binding.robot = "franka"
        binding.robot_cfg = SimpleNamespace(base_pos=(0.0, y, height),
            base_rot=(math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)))
    binding = deepcopy(binding)
    if binding.robot_cfg is None:
        from robobench.robots import FrankaRobotCfg
        binding.robot_cfg = FrankaRobotCfg()
    scene = SCENES.get(source["scene"])(binding.scene_cfg)
    relocate_config_assets(scene.cfg)
    if hasattr(scene.cfg, "grasp_weld"):
        scene.cfg.grasp_weld = False
    binding.scene_cfg = scene.cfg
    return binding, scene


def extra_scene_success_contract(task, backend="gsmini"):
    if backend != "gsmini":
        raise ValueError("Extra scene supports TacMind Franka + GSmini only")
    contract = {"format": "tacmind.extra.scene-goals", "suite": "tac_extra",
            "task_id": task["task_id"], "backend": backend,
            "source": deepcopy(task["source"]), "pose_window_s": 0.0,
            "stable_steps": 10, "stable_duration_s": 0.5,
            "goal": "upstream grader.check_success on finite live physical state",
            "grasp_weld": False}
    goal = task.get("success_config")
    if goal is not None:
        if (task["source"]["scene"] != "coffee"
                or goal.get("goal") != "coffee_brewed_in_cup"):
            raise ValueError(f"Unsupported Extra scene success configuration: {goal}")
        contract.update(format="tacmind.extra.coffee-brewed-goals",
            success_config=deepcopy(goal),
            goal="native brew completed with the filled cup settled under the spout",
            goal_stages=["pod_loaded", "bay_closed", "cup_staged", "started", "brewed"])
    if task["source"]["scene"] == "push_shapes":
        contract.update(format="tacmind.extra.push-shapes-history-goals", require_no_lift=True,
                        sampling="native physics steps",
                        goal="all shapes seated and settled; native board height bounds held throughout the episode")
    return contract


# World-frame eye/target, with Z relative to the authored work surface.
# Keep one fixed observation per mechanism, shared by all seeds and food variants.
AGENTVIEWS = {
    "coffee": ((.75, -.85, .95), (.20, -.015, .17), (0., 0., 1.)),
}


def configure_agentview(cfg, binding, scene):
    """Keep authored task views unless a fixed exception avoids robot occlusion."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    view = scene.CAMERAS.get("front")
    if binding.scene in AGENTVIEWS:
        eye, target, up = map(np.asarray, AGENTVIEWS[binding.scene])
        cfg.scene.agentview.spawn.clipping_range = (.05, 40.)
    elif view is not None:
        eye, target, up = map(np.asarray, (view["eye"], view["target"], (0., 0., 1.)))
        # Upstream replay uses a 20.955 mm aperture; preserve its framing and
        # nominal pose without sampling the optional per-episode camera bands.
        cfg.scene.agentview.spawn.focal_length = view["focal"]
        cfg.scene.agentview.spawn.horizontal_aperture = 20.955
        cfg.scene.agentview.spawn.clipping_range = (.05, 40.)
    else:
        return
    eye = eye + [0., 0., scene.cfg.surface_z]
    target = target + [0., 0., scene.cfg.surface_z]
    forward = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    rotation = Rotation.from_matrix(np.stack((right, np.cross(right, forward), -forward), axis=1))
    q = binding.robot_cfg.base_rot
    base = Rotation.from_quat((*q[1:], q[0]))
    x, y, z, w = (base.inv() * rotation).as_quat()
    cfg.scene.agentview.offset.pos = tuple(base.inv().apply(eye - binding.robot_cfg.base_pos))
    cfg.scene.agentview.offset.rot = (w, x, y, z)


def spawn_native_usd(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Activate authored SDF colliders and honor the upstream mass override."""
    import isaaclab.sim as sim_utils
    from pxr import PhysxSchema, Usd, UsdPhysics
    # IsaacLab's standard USD spawner only modifies an existing MassAPI. Keep
    # its normal reference, materials and contact setup, then define that API
    # on actual bodies using the unchanged upstream mass/density configuration.
    spawn_cfg = deepcopy(cfg)
    spawn_cfg.mass_props = None
    root = sim_utils.spawn_from_usd(prim_path, spawn_cfg, translation, orientation, **kwargs)
    # Some pinned assets author SDF parameters without applying the PhysX API.
    # PhysX otherwise parses them as triangle meshes, causing dynamic convex-
    # hull fallback and default mass/inertia on the kinematic screw bodies.
    # Activate only explicitly selected SDFs; keep every authored parameter.
    for prim in Usd.PrimRange(root):
        mesh = UsdPhysics.MeshCollisionAPI(prim)
        resolution = prim.GetAttribute("physxSDFMeshCollision:sdfResolution")
        if (mesh and mesh.GetApproximationAttr().Get() == "sdf" and resolution
                and resolution.HasAuthoredValueOpinion() and resolution.Get() > 0):
            PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
    if cfg.mass_props is None:
        return root
    bodies = [prim for prim in Usd.PrimRange(root) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    if not bodies:
        raise ValueError(f"Extra scene mass override has no rigid body: {prim_path}")
    for body in bodies:
        sim_utils.define_mass_properties(str(body.GetPath()), cfg.mass_props, stage=root.GetStage())
    return root


class ExtraSceneSuccessMonitor:
    """Read the environment's one per-episode monitor; repeated reads never advance it."""

    def __init__(self, env, task, backend="gsmini"):
        self.contract = extra_scene_success_contract(task, backend)
        if (env.task["task_id"] != task["task_id"] or env.task["source"] != task["source"]
                or env.task.get("success_config") != task.get("success_config")):
            raise ValueError("Extra scene monitor task differs from environment")
        self.env = env

    def update(self, *, advanced=True):
        result = deepcopy(self.env.task_result)
        if result["success"] and not self.env.read_task_state()["success"]:
            result.update(success=False, reason="physical goal no longer holds")
        return result


def make_extra_scene_env_cfg(task, seed=0, *, device="cuda:0", cameras=True, physics_dt=None):
    from isaaclab.envs import mdp
    from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
    from isaaclab.utils import configclass
    from isaaclab.assets import RigidObjectCfg, ArticulationCfg
    from isaaclab.sim import UsdFileCfg
    from benchmark.backends.gsmini.sensors import FrankaGsminiLiftEnvCfg, configure_contact_sensors
    from benchmark.backends.gsmini.environment import _configure_expert_env_cfg
    from benchmark.scene.cameras import configure_rgb_cameras

    binding, scene = task_binding(task)
    sim = scene.sim_cfg()
    # Preserve faster upstream rates and use at least 240 Hz for GSmini contact.
    # Round toward smaller dt while preserving 20 Hz control and observations.
    authored_dt = binding.sim_overrides.get("dt", sim.dt)
    dt = physics_dt if physics_dt is not None else 0.05 / math.ceil(0.05 / min(authored_dt, 1/240) - 1e-10)
    decimation = physics_decimation(dt)
    if dt > authored_dt + 1e-12:
        raise ValueError("Extra scene physics dt cannot be coarser than the authored scene rate")
    cfg = FrankaGsminiLiftEnvCfg()
    _configure_expert_env_cfg(cfg, seed)
    cfg.sim.device, cfg.sim.dt = device, dt
    cfg.decimation = cfg.sim.render_interval = decimation
    cfg.sim.gravity = sim.gravity
    for key, value in {**sim.physx, **binding.sim_overrides.get("physx", {})}.items():
        setattr(cfg.sim.physx, key, value)
    for key, value in {**sim.render, **binding.sim_overrides.get("render", {})}.items():
        setattr(cfg.sim.render, key, value)
    cfg.sim.enable_scene_query_support = True
    cfg.scene.replicate_physics = False
    cfg.scene.robot.init_state.pos = tuple(binding.robot_cfg.base_pos)
    cfg.scene.robot.init_state.rot = tuple(binding.robot_cfg.base_rot)
    cfg.scene.object = cfg.scene.table = cfg.scene.plane = cfg.scene.light = None
    cfg.events.reset_robot_joints = None
    cfg.terminations.object_dropping = cfg.terminations.success = None
    cfg.episode_length_s = task["episode_config"]["timeout_s"]

    @configclass
    class RobotObservationsCfg(ObservationGroupCfg):
        joint_positions = ObservationTermCfg(func=mdp.joint_pos)
        joint_velocities = ObservationTermCfg(func=mdp.joint_vel)
        enable_corruption = False
        concatenate_terms = True

    cfg.observations.policy = RobotObservationsCfg()
    assets = scene.assets()
    relocate_config_assets(assets)
    from benchmark.tasks.extra.scene_packing_assets import configure_assets
    configure_assets(assets, task)
    from benchmark.tasks.extra.scene_materials import configure_assets as configure_materials
    configure_materials(assets)
    if task["source"]["scene"] in {"classify_objects", "stack_blocks"}:
        # These color-coded primitives have no USD material layer. Keep their
        # colors and physical materials while refining the opaque finish.
        for name, asset in assets.items():
            material = getattr(asset.spawn, "visual_material", None)
            if material is not None:
                asset.spawn.visual_material = deepcopy(material)
                asset.spawn.visual_material.roughness = (
                    .72 if name.startswith("zone_") or name == "pad" else .44)
    reserved = {"robot", "ee_frame", "agentview", "wristview", "gsmini_left", "gsmini_right"}
    if reserved.intersection(assets):
        raise ValueError(f"Upstream scene shadows TacMind components: {reserved.intersection(assets)}")
    filters = []
    for name, asset in assets.items():
        # A single native scene runs in env_0. A broad env_.* regex can cross
        # '/' and match GSmini's Looks/Plate material before the physical Plate.
        asset.prim_path = asset.prim_path.replace("{ENV_REGEX_NS}","/World/envs/env_0")
        asset.prim_path = asset.prim_path.replace("/World/envs/env_.*","/World/envs/env_0")
        if name == "ground" and getattr(asset.spawn, "usd_path", None):
            from benchmark.tasks.extra.scene_assets import SOURCE, asset_path
            asset.spawn.usd_path = str(asset_path(
                SOURCE / "robobench/suites/assembly/assets/props/ground/default_ground.usd"))
        if getattr(asset.spawn, "usd_path", None):
            from benchmark.tasks.extra.scene_assets import localized_usd
            from benchmark.tasks.extra.scene_materials import enhanced_usd
            source_usd = asset.spawn.usd_path
            asset.spawn.usd_path = localized_usd(source_usd)
            asset.spawn.usd_path = enhanced_usd(source_usd, asset.spawn.usd_path)
            if isinstance(asset.spawn, UsdFileCfg):
                asset.spawn.func = spawn_native_usd
        if isinstance(asset, (RigidObjectCfg, ArticulationCfg)) and asset.spawn is not None:
            asset.spawn.activate_contact_sensors = True
            filters.extend(_body_filter_paths(asset))
        setattr(cfg.scene, name, asset)
    if task.get("background"):
        from benchmark.scene.visuals import task_background_path
        cfg.scene.light.spawn.texture_file = str(task_background_path(task))
        cfg.scene.light.spawn.texture_format = "latlong"
        cfg.scene.light.spawn.visible_in_primary_ray = True
    configure_contact_sensors(cfg, filters)
    configure_rgb_cameras(cfg.scene, suite="tac_extra")
    configure_agentview(cfg, binding, scene)
    # The learning contract uses the same image shape for both RGB streams.
    cfg.scene.wristview.width = cfg.scene.agentview.width
    cfg.scene.wristview.height = cfg.scene.agentview.height
    cfg.sim.render.antialiasing_mode = "FXAA"
    if not cameras:
        cfg.scene.agentview = cfg.scene.wristview = None
        cfg.observations.vision = None
    cfg.extra_scene_asset_names = tuple(assets)
    cfg.extra_scene_authored_dt = authored_dt
    return cfg, binding, scene


def _body_filter_paths(asset):
    """Resolve nested USD bodies, rather than filtering a nonphysical wrapper."""
    from pxr import Usd, UsdPhysics
    path = getattr(asset.spawn, "usd_path", None)
    if not path:
        return [asset.prim_path]
    stage = Usd.Stage.Open(path)
    root = stage.GetDefaultPrim() if stage else None
    if not root:
        raise ValueError(f"Extra scene USD has no default prim: {path}")
    return [asset.prim_path + ("/" + str(prim.GetPath().MakeRelativePath(root.GetPath()))
            if prim != root else "") for prim in Usd.PrimRange(root)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)]




class ExtraSceneEnvironment:
    """One environment factory for preview, recording and policy evaluation.

    Uses ManagerBasedEnv stepping, avoiding RL auto-reset during long tasks.
    The upstream facade exposes its scene separately from TacMind's asset map.
    """

    episode = None
    step_observer = None

    def __init__(self, task_id, seed=0, *, device="cuda:0", cameras=True, physics_dt=None,
                 camera_shapes=None):
        from isaaclab.envs import ManagerBasedEnv
        from isaaclab.sim import create_new_stage
        self.task = get_task("tac_extra", task_id)
        from benchmark.catalog import is_extra_scene
        if not is_extra_scene(self.task):
            raise ValueError("This task uses the shared Extra environment; construct ExtraEnvironment instead")
        self._resetting, self._closed = True, False
        cfg, self.binding, self.upstream_scene = make_extra_scene_env_cfg(
            self.task, seed, device=device, cameras=cameras, physics_dt=physics_dt)
        if camera_shapes is not None:
            from benchmark.scene.cameras import configure_camera_shapes
            configure_camera_shapes(cfg.scene, camera_shapes)
        create_new_stage()
        self.env = ManagerBasedEnv(cfg)
        self._original_update = self.env.scene.update
        try:
            self.facade = SimpleNamespace(iscene=self.env.scene, scene=self.upstream_scene,
                sim=self.env.sim, stage=self.env.sim.stage, num_envs=self.env.num_envs,
                device=self.env.device, dt=cfg.sim.dt,
                robot=SimpleNamespace(articulation=self.env.scene["robot"], cfg=self.binding.robot_cfg,
                                      control_period=cfg.decimation))
            self.upstream_scene.bind(self.facade)
            self.env.scene.update = self._update_scene
            from robobench.core.rooms import prepare_room, spawn_room
            room, _ = prepare_room(self.binding, self.upstream_scene, self.facade.robot)
            if room is not None:
                from benchmark.tasks.extra.scene_assets import localized_usd
                from benchmark.tasks.extra.scene_materials import enhanced_usd
                source_usd = room["room"]["usd"]
                room["room"]["usd"] = enhanced_usd(source_usd, localized_usd(source_usd))
                spawn_room(self.facade, room)
            self.room = room
            self.reset(seed=seed)
        except BaseException:
            self.close()
            raise

    def __getattr__(self, name):
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)

    @property
    def common_step_counter(self):
        return self.control_steps

    @property
    def diagnostic_object(self):
        return (self.upstream_scene.knife if hasattr(self.upstream_scene, "knife")
                else next(iter(self.scene.rigid_objects.values())))

    def _update_scene(self, dt, *args, **kwargs):
        result = self._original_update(dt, *args, **kwargs)
        if dt > 0 and not self._resetting:
            self.upstream_scene.post_step()
            self.physics_steps += 1
            if self.task["source"]["scene"] == "push_shapes":
                import torch
                scene = self.upstream_scene
                self.board_path_valid &= torch.stack([
                    scene.not_lifted(name) for name in scene.blocks]).all(dim=0)
                index = (self.physics_steps - 1) % len(self.board_step_history)
                self.board_height_history[index] = torch.stack([
                    scene.block_bottom_z(name) for name in scene.blocks], dim=-1)[0]
                self.board_step_history[index] = self.physics_steps
        return result


    def reset(self, *, seed=0, options=None):
        import torch
        if type(seed) is not int or not 0 <= seed < 2**32 or options:
            raise ValueError("Extra scene reset accepts an integer seed in [0, 2**32) only")
        self._resetting = True
        self.episode = None
        self.env.reset(seed=seed)
        ids = torch.arange(self.num_envs, device=self.device)
        self.upstream_scene.reset(ids)
        for sensor in self.scene.sensors.values():
            sensor.reset(ids)
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(0.0)
        self.physics_steps = self.control_steps = self.stable_steps = 0
        self.contact_seen = False
        if self.task["source"]["scene"] == "push_shapes":
            self.board_path_valid = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            self.board_height_history = torch.zeros(
                (self.cfg.decimation, len(self.upstream_scene.blocks)), device=self.device)
            self.board_step_history = torch.zeros(self.cfg.decimation, dtype=torch.int64, device=self.device)
        self.action = torch.zeros_like(self.action_manager.action)
        from benchmark.backends.gsmini.environment import _hold_current_arm_action
        _hold_current_arm_action(self, self.action)
        self.action[:, -1] = 1.0
        self.task_result = {"success": False, "score": 0.0, "reason": "reset; no physical interaction"}
        registry = import_module(f"robobench.suites.{self.task['source']['suite']}.grader").GRADERS
        self.grader = registry[self.task["source"]["scene"]](self.facade)
        self._resetting = False
        for _ in range(3):
            self.sim.render()
        self.env.obs_buf = self.observation_manager.compute()
        return self.env.obs_buf, {"suite": "tac_extra", "task_id": self.task["task_id"], "seed": seed}

    def read_task_state(self):
        """Reject invalid mechanics before an upstream grader can clamp NaN scores."""
        import torch
        for group in (self.scene.rigid_objects, self.scene.articulations):
            for name, asset in group.items():
                state = asset.data.root_state_w
                if not torch.isfinite(state).all() or (torch.linalg.vector_norm(state[:, 3:7], dim=-1) < 1e-7).any():
                    raise RuntimeError(f"Non-finite or degenerate Extra scene state: {name}")
                if hasattr(asset.data, "joint_pos") and not all(torch.isfinite(value).all() for value in
                        (asset.data.joint_pos, asset.data.joint_vel)):
                    raise RuntimeError(f"Non-finite Extra scene articulation: {name}")
        values = self.grader.measure()
        score = self.grader.progress(values)
        success = torch.as_tensor(self.grader.check_success())
        if (score.numel() != 1 or success.numel() != 1 or success.dtype != torch.bool
                or not torch.isfinite(score).all()
                or any(value.numel() != 1 or not torch.isfinite(value).all() for value in values.values())):
            raise RuntimeError("Invalid Extra scene grader result")
        if self.task.get("success_config") is not None:
            contract = extra_scene_success_contract(self.task)
            values = {name: values[name] for name in contract["goal_stages"]}
            completed = torch.stack(tuple(values.values()))
            score = completed.float().mean(dim=0)
            scene = self.upstream_scene
            received = scene.filled() & scene.cup_under_spout() & scene.cup_settled()
            success = completed.bool().all(dim=0).to(received.device) & received
            if success.numel() != 1 or success.dtype != torch.bool:
                raise RuntimeError("Invalid Extra scene coffee goal result")
        history = {}
        if self.task["source"]["scene"] == "push_shapes":
            success &= self.board_path_valid
            history["board_path_valid"] = bool(self.board_path_valid.item())
            if not history["board_path_valid"]:
                history["reason"] = "a shape left the permitted board height range during the episode"
        return {"success": bool(success.item()), "score": float(score.item()), **history,
                "stages": {name: float(value[0]) for name, value in values.items()}}

    def step(self, action):
        import torch
        if action.shape != self.action.shape or not torch.isfinite(action).all():
            raise ValueError(f"Extra scene action must be finite with shape {tuple(self.action.shape)}")
        if (not getattr(self, "policy_evaluation", False)
                and self.control_steps * self.step_dt >= self.task["episode_config"]["timeout_s"]):
            raise RuntimeError("Extra scene episode timed out; reset required")
        self.action.copy_(action)
        from benchmark.backends.gsmini.recording import _prepare_recording_step
        _prepare_recording_step(self, self.action)
        observations, info = self.env.step(self.action)
        self.control_steps += 1
        sample = self.read_task_state()
        self.grader.record()
        for side in ("left", "right"):
            sensor = self.scene[f"gsmini_{side}_contact"]
            if not torch.isfinite(sensor.data.force_matrix_w).all():
                raise RuntimeError(f"Non-finite Extra scene {side} gel forces")
            self.contact_seen |= bool((sensor.data.force_matrix_w.norm(dim=-1) > 0.05).any())
        self.stable_steps = self.stable_steps + 1 if sample["success"] else 0
        success = sample["success"] and self.stable_steps >= 10 and self.contact_seen
        timed_out = (not getattr(self, "policy_evaluation", False)
                     and self.control_steps * self.step_dt >= self.task["episode_config"]["timeout_s"])
        self.task_result = {**sample, "success": success and not timed_out,
                            "stable_steps": self.stable_steps,
                            "reason": sample.get("reason", "goal reached" if success else "goal or stability window incomplete")}
        if self.step_observer is not None:
            self.step_observer()
        return (observations, torch.zeros(1, device=self.device),
                torch.tensor([success and not timed_out], device=self.device),
                torch.tensor([timed_out], device=self.device), {**info, "task_success": self.task_result})

    def close(self):
        if not self._closed:
            self._closed = True
            self.env.scene.update = self._original_update
            self.env.close()


def state_snapshot(env):
    """Small independent CPU tensors for reset repeatability and command traces."""
    result = {}
    for name, asset in {**env.scene.rigid_objects, **env.scene.articulations}.items():
        result[name] = asset.data.root_state_w.detach().cpu().tolist()
        if hasattr(asset.data, "joint_pos"):
            result[name + "/joints"] = asset.data.joint_pos.detach().cpu().tolist()
    return result
