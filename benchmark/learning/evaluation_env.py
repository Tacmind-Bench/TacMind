"""Policy rollout environments reuse the collectors' task, reset and physics contracts."""

from contextlib import ExitStack, contextmanager
from functools import partial

from benchmark.catalog import is_extra_scene, is_tabletop_task


class PolicyEvaluationSession:
    """Keep one supported task's native scene and optical caches across seeds."""

    def __init__(self, app, task, suite, spec, recorded_contract, *, perturbation=None):
        self.app, self.task, self.suite = app, task, suite
        self.spec, self.recorded_contract = spec, recorded_contract
        self.perturbation = perturbation
        self.resources = ExitStack()
        self.env = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            return self.resources.__exit__(*exc)
        finally:
            self.env = None

    @contextmanager
    def episode(self, seed):
        env = None
        extra = self.suite == "tac_extra" and not is_tabletop_task(self.task)
        try:
            with policy_environment(self.app, self.task, self.suite, seed, self.spec,
                                    self.recorded_contract, perturbation=self.perturbation,
                                    _session=self) as environment:
                env = environment[0]
                if extra:
                    env.policy_evaluation = True
                yield environment
        finally:
            if env is not None and extra:
                env.policy_evaluation = False
            if self.env is not None:
                self.env.policy_success_monitor = None

    def reset(self, seed):
        env = self.env
        # Native reset clears actions, manager histories and sensor state. Unbind
        # the previous monitor before it invokes PolicyEvaluationMixin._reset_idx.
        env.policy_success_monitor = None
        env.policy_terminal_goal_status = None
        env.cfg.seed = seed
        if self.suite == "tac_extra" and not is_tabletop_task(self.task):
            # Extra owns its counters, contact history and any cut joints. Its
            # reset also returns the task's prepared initial action.
            env.reset(seed=seed)
            return env.action
        env.cfg.episode_length_s = self.initial_episode_length_s
        env.common_step_counter = env._sim_step_counter = 0
        if self.suite == "tac_dynamic":
            from benchmark.tasks.dynamic.scene import prepare_dynamic_reset
            reset, paths = prepare_dynamic_reset(env, seed)
            env.reset(seed=seed)
            return _prepare_dynamic_episode(self.resources, env, self.app, self.task, seed, reset, paths)
        env.reset(seed=seed)
        return prepare_tabletop_episode(env, self.app, self.task, self.suite, seed)


class PolicyEvaluationMixin:
    """Preserve the final physical goal state across IsaacLab's automatic reset."""

    def start_policy_episode(self, monitor, max_steps):
        initialization_steps = int(self.episode_length_buf[0].item())
        self.episode_length_buf.zero_()
        self.cfg.episode_length_s = max_steps * self.step_dt
        self.policy_success_monitor = monitor
        self.policy_terminal_goal_status = None
        return initialization_steps

    def _reset_idx(self, env_ids):
        monitor = getattr(self, "policy_success_monitor", None)
        if monitor is not None and len(env_ids):
            self.policy_terminal_goal_status = monitor.update()
        return super()._reset_idx(env_ids)


def policy_task(suite, task_id):
    from benchmark.catalog import get_task

    task = get_task(suite, task_id)
    if is_tabletop_task(task):
        from benchmark.expert.tabletop import resolve_tabletop_task
        return resolve_tabletop_task(task)
    if suite == "tac_dynamic":
        from benchmark.tasks.dynamic.scene import resolve_gsmini_task
        return resolve_gsmini_task(task)
    return task


def prepare_tabletop_episode(env, app, task, suite, seed):
    from benchmark.backends.gsmini.environment import reset_tabletop_task
    from benchmark.backends.gsmini.execution import hold

    in_hand = (task.get("reset_contract") or {}).get("mode") == "in_hand"
    options = {}
    if is_tabletop_task(task):
        from benchmark.expert.tabletop import tabletop_joint_defaults, settle_tabletop_scene
        options["joint_defaults"] = tabletop_joint_defaults
        settle = settle_tabletop_scene
    else:
        from benchmark.backends.gsmini.execution import settle_scene as _settle_scene
        settle = _settle_scene
    action = reset_tabletop_task(env, task, suite=suite, seed=seed, allow_in_hand=in_hand, **options)
    if in_hand:
        from benchmark.expert.tabletop import _initialize_in_hand
        env.sim.forward()
        env.scene.update(0.0)
        _initialize_in_hand(env, task)
        action[:, -1] = -1.0
        hold(env, app, action, 0.5)
    else:
        action[:, -1] = 1.0
        settle(env, app, action, task)
    return action


def _prepare_dynamic_episode(resources, env, app, task, seed, reset, paths):
    import torch
    from benchmark.catalog import ASSETS, ROBOT_ASSET
    from benchmark.tasks.dynamic.scene import dynamic_payload_task, _belt_range, _dynamic_obstacle_paths, _verify_source_seating
    from benchmark.backends.gsmini.environment import _hold_current_arm_action
    from benchmark.expert.tabletop import _initialize_in_hand
    from benchmark.backends.gsmini.execution import hold

    episode = task["episode_config"]
    in_hand = episode.get("in_hand_reset", {}).get("mode") == "in_hand"
    threaded = in_hand and episode["object_family"] == "bulb"
    action = torch.zeros_like(env.action_manager.action)
    _hold_current_arm_action(env, action)
    if in_hand:
        synthetic = dynamic_payload_task(task)
        env.task = synthetic
        env.sim.forward()
        env.scene.update(0.0)
        _initialize_in_hand(env, synthetic)
        close = float(episode["in_hand_reset"].get("preclose_command", -0.7))
        action[:, -1] = close if threaded else -1.0
        hold(env, app, action, 0.5)
    contract = episode.get("in_hand_reset", {})
    command = float(contract.get("gelsight_transport_command",
                    contract.get("gelsight_hold_command", contract.get("preclose_command", -0.7))))
    action[:, -1] = (command if threaded else -1.0) if in_hand else 1.0
    from benchmark.tasks.dynamic.scene import settle_dynamic_scene
    settle_dynamic_scene(env, app, action)
    if paths["source_holder"]:
        poses = {role: (env.scene[role].data.root_pos_w[0].cpu().tolist(),
                        env.scene[role].data.root_quat_w[0].cpu().tolist())
                 for role in ("object", "source_holder")}
        _verify_source_seating(env.sim.stage, paths, poses)
    platform = env.scene["platform"].data
    clearance = (platform.root_pos_w[0, 2].item() - float(episode["platform_size_m"][2]) / 2
                 - float(_belt_range(env.sim.stage)[1][2]))
    moving = env.scene["fixture" if paths["fixture"] == "/World/PreviewMovingAsset" else "object"].data
    expected_speed = reset["belt_speed_mps"]
    if (abs(clearance) > 0.005 or abs(platform.root_vel_w[0, 1].item()) < expected_speed * 0.35
            or abs(moving.root_vel_w[0, 1].item()) < expected_speed * 0.25):
        raise RuntimeError("Dynamic evaluation platform/asset did not settle and move with the physical belt")
    if in_hand:
        from benchmark.backends.gsmini.motion import _create_motion_planner
        from benchmark.tasks.dynamic.gsmini import _wait_for_dynamic_target
        from benchmark.backends.gsmini.sensors import planner_robot
        from benchmark.backends.gsmini.workspace import read_collision_world

        obstacles = _dynamic_obstacle_paths(paths)
        def world(names):
            return read_collision_world(env, obstacle_paths=tuple(obstacles[name] for name in names),
                                        robot_link_path="/World/FrankaArm/panda_link0")
        planner = resources.enter_context(_create_motion_planner(
            env, seed, world(name for name in obstacles if name not in ("object", "fixture")),
            world(obstacles), robot_config=partial(planner_robot, robot_usd_path=str(ASSETS / ROBOT_ASSET)),
            velocity_scale=0.5))
        planner.warmup(enable_graph=True, num_warmup_iterations=1)
        _wait_for_dynamic_target(env, app, planner, action, task)
    return action


@contextmanager
def policy_environment(app, task, suite, seed, spec, recorded_contract, *, perturbation=None,
                       _session=None):
    if suite == "tac_extra" and perturbation is not None and perturbation.enabled:
        raise ValueError("ExtraSuite does not support evaluation perturbations; omit --perturb")
    if _session is not None and _session.env is not None:
        action = _session.reset(seed)
        yield _session.env, action
        return
    camera_shapes = {name: spec.camera_shape(name)
                     for name in (*spec.camera_names, "gsmini_left", "gsmini_right")}
    if suite == "tac_extra" and not is_tabletop_task(task):
        from benchmark.tasks.extra.environment import ExtraEnvironment
        from benchmark.backends.gsmini.execution import configure_shutdown
        if not is_extra_scene(task):
            from isaaclab.sim import create_new_stage
            create_new_stage()
        physics_dt = recorded_contract.get("physics_dt", recorded_contract.get("simulation", {}).get("physics_dt"))
        env = ExtraEnvironment(task["task_id"], seed=seed, physics_dt=physics_dt,
                               camera_shapes=camera_shapes)
        with ExitStack() as resources:
            resources.callback(env.close)
            configure_shutdown(env)
            resources.callback(env.sim.stop)
            env.evaluation_perturbation = {}
            if _session is not None:
                _session.env = env
                _session.resources = resources.pop_all()
            yield env, env.action
        return
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sim import create_new_stage
    from benchmark.backends.gsmini.environment import initialize_task_environment, _make_env_cfg
    from benchmark.backends.gsmini.execution import configure_shutdown

    class PolicyEvaluationEnv(PolicyEvaluationMixin, ManagerBasedRLEnv):
        pass

    from benchmark.scene.perturbations import EvalPerturbationSpec
    from benchmark.learning.perturbation_scene import evaluation_lighting, configure_eval_camera
    from benchmark.scene.perturbations import _look_at_on_plane
    from benchmark.scene.cameras import camera_metadata

    look = perturbation if perturbation is not None else EvalPerturbationSpec()
    with ExitStack() as resources:
        if suite != "tac_dynamic":
            create_new_stage()
        if suite == "tac_dynamic":
            from benchmark.tasks.dynamic import scene as dynamic
            reset = dynamic.sample_dynamic_reset(task, seed)
            stage = dynamic._load_dynamic_stage(app, task)
            paths = dynamic._prepare_dynamic_scene(stage, task, reset,
                drop_clearance_m=float(task["episode_config"].get("platform_drop_clearance_m", 0.01)))
            dynamic.configure_conveyor_motor(stage, reset["belt_speed_mps"])
            cfg = dynamic._make_env_cfg(seed, task, stage, paths)
        elif is_tabletop_task(task):
            from benchmark.expert.tabletop import make_tabletop_environment_config
            cfg = make_tabletop_environment_config(seed, task)
        else:
            cfg = _make_env_cfg(seed, task)
        from benchmark.scene.cameras import configure_camera_shapes
        configure_camera_shapes(cfg.scene, camera_shapes)
        orbit = None
        plane_z = (float(cfg.scene.platform.init_state.pos[2]) +
                   float(task["episode_config"]["platform_size_m"][2]) / 2 -
                   float(cfg.scene.robot.init_state.pos[2]) if suite == "tac_dynamic" else
                   float(task["workspace_offset"][2]) - float(cfg.scene.robot.init_state.pos[2]))
        if look.enabled:
            orbit = configure_eval_camera(cfg.scene, task, suite, look, look_at_z=plane_z)
        env = PolicyEvaluationEnv(cfg=cfg)
        initialize_task_environment(resources, env, seed, configure_shutdown, task=task)
        resources.callback(env.sim.stop)
        lighting = None
        if look.lighting:
            camera = env.scene["agentview"].data
            eye = tuple(camera.pos_w[0].cpu().tolist())
            rotation = tuple(camera.quat_w_opengl[0].cpu().tolist())
            target = _look_at_on_plane(eye, rotation, plane_z + float(cfg.scene.robot.init_state.pos[2]))
            lighting = resources.enter_context(evaluation_lighting(
                env.sim.stage, look, authored_lights=bool(task.get("scene_layout")),
                camera_position=eye, look_at=target))
        geometry = {"objects": {name: {key: value for key, value in info.items() if key in ("type", "scale")}
                                for name, info in task.get("objects", {}).items()}}
        if suite == "tac_dynamic":
            geometry["episode"] = {key: value for key, value in task["episode_config"].items()
                if key.endswith(("_scale", "_mass_kg", "_density")) or key in
                ("platform_size_m", "goal_offset_in_fixture_m", "source_payload_offset_m")}
        env.evaluation_perturbation = {**look.to_dict(), "agentview_orbit": orbit,
            "geometry": geometry,
            "lighting_runtime": lighting, "cameras": camera_metadata(env.cfg.scene)}
        if suite == "tac_dynamic":
            dynamic.bind_dynamic_reset(env, task, reset, paths)
        action = (_prepare_dynamic_episode(resources, env, app, task, seed, reset, paths)
                  if suite == "tac_dynamic" else prepare_tabletop_episode(env, app, task, suite, seed))
        from benchmark.expert.capabilities import REUSABLE_GSMINI_TASK_IDS

        if (_session is not None and
                task["task_id"] in REUSABLE_GSMINI_TASK_IDS.get(suite.removeprefix("tac_"), ())):
            _session.env = env
            _session.initial_episode_length_s = env.cfg.episode_length_s
            _session.resources = resources.pop_all()
        yield env, action
