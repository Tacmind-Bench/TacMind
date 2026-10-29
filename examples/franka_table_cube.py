# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the Franka cube-grasp example with Panda or GSmini.

    conda activate tacmind
    source third_party/IsaacLab/_isaac_sim/setup_conda_env.sh
    python -m examples.franka_table_cube --robot panda --num-envs 4

Requires Isaac Sim 5.1 / Isaac Lab 2.3, cuRobo's MotionPlanner API (0.8+),
and warp-lang==1.10.1 (shared by Isaac Sim and cuRobo).
Uses benchmark.backends.gsmini.robot.FrankaLiftEnvCfg for shared, effort-limited gravity compensation.
Each environment plans independently and executes in the same physics steps.
Identical table layouts share one robot-local collision world.
Reset samples reachable tabletop positions; --seed reproduces them and
--workspace-margin sets boundary clearance in meters (default 0.05).
The object then falls onto the table; planning waits for stable physical rest.
Add --headless to exit after all grasps have been verified.
In GUI mode, closing the window or pressing Stop exits; Pause remains responsive.
An interrupted headless run exits with an error.
"""

import argparse
from contextlib import ExitStack
from pathlib import Path


from benchmark.backends.gsmini.execution import simulation_app, configure_shutdown, settle_object, plan_grasp, execute_trajectory, hold, verify_grasp, step_env

def run(args: argparse.Namespace, app, cfg=None, on_complete=None, *,
        env_class=None, export_plan=None) -> None:
    """Settle and plan; export the plan or execute and validate the grasp."""
    import torch
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.batch_motion_planner import BatchMotionPlanner
    from curobo.types import DeviceCfg
    from isaaclab.envs import ManagerBasedRLEnv
    from benchmark.backends.gsmini.robot import FrankaLiftEnvCfg
    from benchmark.backends.gsmini.workspace import read_collision_world

    cfg = FrankaLiftEnvCfg() if cfg is None else cfg
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    planner_robot = cfg.events.reset_object_position.params.get("robot_config")
    if hasattr(args, "workspace_margin"):
        cfg.events.reset_object_position.params["margin"] = args.workspace_margin
    cfg.terminations.time_out = None
    cfg.commands.object_pose.debug_vis = False
    cfg.observations.policy.enable_corruption = False
    cfg.viewer.eye = (2.0, -2.0, 1.5)
    cfg.viewer.lookat = (0.4, 0.0, 0.2)
    with ExitStack() as resources:
        env = (ManagerBasedRLEnv if env_class is None else env_class)(cfg=cfg)
        resources.callback(env.close)
        configure_shutdown(env)
        env.reset()
        print(f"[OBJECT] seed={args.seed}, positions="
              f"{(env.scene['object'].data.root_pos_w - env.scene.env_origins).tolist()}", flush=True)
        action = torch.zeros_like(env.action_manager.action)
        action[:, -1] = 1.0  # Fully open the continuous gripper.
        settle_object(env, app, action)
        initial_z = env.scene["object"].data.root_pos_w[:, 2].clone()
        class RetryingBatchMotionPlanner(BatchMotionPlanner):
            def plan_pose(self, *args, **kwargs):
                kwargs.setdefault("max_attempts", 5)
                return super().plan_pose(*args, **kwargs)

        planner_type = RetryingBatchMotionPlanner if env.num_envs > 1 else MotionPlanner
        planner = planner_type(MotionPlannerCfg.create(
            robot="franka.yml" if planner_robot is None else planner_robot(env), scene_model=read_collision_world(env),
            device_cfg=DeviceCfg(device=env.device), interpolation_dt=env.step_dt,
            random_seed=args.seed, max_batch_size=env.num_envs,
        ))
        resources.callback(planner.destroy)
        planner.warmup(enable_graph=True)
        plan = plan_grasp(env, planner)
        if export_plan is not None:
            export_plan(env, plan)
            return
        settle_seconds = getattr(cfg, "trajectory_settle_seconds", 1.0)
        execute_trajectory(env, app, plan, "approach", action, settle_seconds=settle_seconds)
        execute_trajectory(env, app, plan, "grasp", action, settle_seconds=settle_seconds)
        tcp = env.scene["ee_frame"].data.target_pos_w[:, 0]
        distance = torch.linalg.vector_norm(tcp - env.scene["object"].data.root_pos_w, dim=-1)
        if not ((distance >= 0.0) & (distance <= 0.025)).all():
            raise RuntimeError("TCP did not reach the cube before closure")
        action[:, -1] = -1.0
        hold(env, app, action, getattr(cfg, "closure_hold_seconds", 0.5))
        execute_trajectory(env, app, plan, "lift", action, settle_seconds=settle_seconds)
        verify_grasp(env, app, action, initial_z)
        if on_complete is not None:
            on_complete(env)
        while not args.headless and app.is_running():
            step_env(env, app, action)


def parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Parse options shared by the Franka gripper examples."""
    from isaaclab.app import AppLauncher

    parser.add_argument("--robot", choices=("panda", "gsmini"),
                        default="panda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workspace-margin", type=float, default=0.05, help="Reachability boundary clearance in meters")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel grasp environments")
    parser.add_argument("--output-dir", type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    return validate_args(args, parser)


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser):
    if args.robot not in ("panda", "gsmini"):
        parser.error(f"Unknown robot: {args.robot}; choose panda or gsmini")
    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    if args.device not in ("cuda", "cuda:0"):
        parser.error("This cuRobo example requires --device cuda:0")
    if args.robot == "panda" and args.output_dir is not None:
        parser.error("Sensor output options require a tactile robot")
    if args.robot != "panda":
        args.output_dir = args.output_dir or (
            Path(__file__).resolve().parents[1] / "tmp" / args.robot
        )
    return args


def save_tactile(env, output_dir):
    """Save GelSight Mini images and depth from the completed grasp."""
    import numpy as np
    import torch
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    for side in ("left", "right"):
        sensor = env.scene[f"gsmini_{side}"]
        rgb = sensor.data.output["tactile_rgb"]
        depth = sensor.camera.data.output["depth"]
        if rgb.shape != (env.num_envs, 240, 320, 3) or rgb.dtype != torch.uint8:
            raise RuntimeError(f"Invalid {side} tactile RGB")
        if not torch.isfinite(depth).flatten(1).any(dim=1).all():
            raise RuntimeError(f"{side} camera has no finite depth samples")
        rgb = rgb.cpu().numpy()
        depth = depth.cpu().numpy()
        for index in range(env.num_envs):
            Image.fromarray(rgb[index]).save(output_dir / f"env_{index}_{side}.png")
            np.save(output_dir / f"env_{index}_{side}_depth.npy", depth[index])
        print(f"[TACTILE] {side}: RGB {tuple(rgb.shape)}, saved to {output_dir}", flush=True)


def main() -> None:
    # Preload installed Warp before Kit adds its older bundled version.
    import warp  # noqa: F401
    from isaaclab.app import AppLauncher

    args = parse_args(argparse.ArgumentParser(description=__doc__))
    args.enable_cameras = args.robot != "panda"
    with simulation_app(AppLauncher(args), require_completion=args.headless) as app:
        if args.robot == "panda":
            run(args, app)
        elif args.robot == "gsmini":
            from benchmark.backends.gsmini.sensors import FrankaGsminiLiftEnvCfg

            run(args, app, cfg=FrankaGsminiLiftEnvCfg(),
                on_complete=lambda env: save_tactile(env, args.output_dir))


if __name__ == "__main__":
    main()
