"""Evaluate policies using the checkpoint's recorded simulation backend."""

import argparse
from contextlib import ExitStack
from functools import lru_cache
import json
from pathlib import Path
import sys
from time import monotonic

from policy import EXTERNAL_TYPES
from benchmark.catalog import ROOT
from benchmark.learning.reporting import evaluation_report

from benchmark.io import write_json
from benchmark.learning.provenance import checkpoint_metadata


@lru_cache(maxsize=1)
def _evaluation_lengths():
    return json.loads((ROOT / "benchmark/config/gsmini_evaluation_lengths.json").read_text())


def evaluation_max_steps(suite, task_id):
    data = _evaluation_lengths()
    return data["tasks"][suite][task_id]["max_recorded_steps"] + data["margin_steps"]


def evaluation_seeds(args):
    if args.training_seeds is not None:
        seeds = list(args.training_seeds)
        if not seeds:
            raise ValueError("Evaluation requires at least one seed")
        if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
            raise ValueError("Evaluation seeds must be unsigned 32-bit integers")
        return seeds
    if type(args.episodes) is not int or args.episodes <= 0:
        raise ValueError("Evaluation episodes must be a positive integer")
    if (type(args.seed_start) is not int or args.seed_start < 0
            or args.seed_start + args.episodes > 2**32):
        raise ValueError("Evaluation seeds must be unsigned 32-bit integers")
    return list(range(args.seed_start, args.seed_start + args.episodes))


def resolve_checkpoint_options(args, training, metadata):
    """Resolve CLI defaults from saved training settings before loading a runtime."""
    kind = training["policy"]["type"]
    if args.policy_type is None:
        args.policy_type = kind
    if args.suite is None:
        args.suite = metadata["suite"]
    if args.task_id is None:
        args.task_id = metadata["task_id"]
    args.max_steps = evaluation_max_steps(args.suite, args.task_id)
    if kind in EXTERNAL_TYPES:
        from policy.checkpoint import load_checkpoint
        saved = load_checkpoint(args.checkpoint)["config"]
    else:
        saved = json.loads((args.checkpoint / "config.json").read_text())
    if args.temporal_agg is None:
        args.temporal_agg = kind == "act"
    if args.temporal_coeff is None:
        args.temporal_coeff = saved.get("temporal_ensemble_coeff") if kind == "act" else .01
    if args.training_seeds is not None:
        args.episodes = len(args.training_seeds)
    return saved


class EvaluationRun:
    """Own a loaded policy across perturbation looks."""

    def __init__(self, args, *, provenance=None):
        self.checkpoint = Path(args.checkpoint).expanduser().resolve()
        self.provenance = checkpoint_metadata(self.checkpoint) if provenance is None else provenance
        self.runner = None
        self.policy_report = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        runner, self.runner = self.runner, None
        if hasattr(runner, "close"):
            runner.close()

    def policy(self, args, spec, metadata):
        if self.runner is None:
            from benchmark.learning.evaluation_policy import load_evaluation_policy
            self.runner, self.policy_report = load_evaluation_policy(args, spec, metadata)
        return self.runner, self.policy_report


def evaluate(app, args, report, *, run=None):
    with ExitStack() as resources:
        if run is None:
            run = resources.enter_context(EvaluationRun(args))
        _, _, metadata = run.provenance
        args.max_steps = evaluation_max_steps(args.suite, args.task_id)
        seeds = evaluation_seeds(args)
        look = getattr(args, "perturbation", None)
        diagnostic = args.training_seeds is not None
        report.update(status="running", episodes=[], episodes_completed=0, successes=0, score_complete=False,
                      suite=args.suite, task_id=args.task_id, backend=metadata.get("backend", "gsmini"),
                      checkpoint=str(run.checkpoint), episodes_requested=len(seeds), seeds=seeds,
                      seed_start=None if diagnostic else args.seed_start, max_steps=args.max_steps,
                      score_type="diagnostic" if diagnostic else "robustness" if look and look.enabled else "nominal")
        report.pop("error", None)
        report.pop("success_rate", None)
        report.pop("completed_episode_success_rate", None)
        with evaluation_report(report, args.output_dir / "results.json"):
            _evaluate_episodes(app, args, report, run, seeds)


def _evaluate_episodes(app, args, report, run, seeds):
    """Collect episode outcomes; report publication is owned by the caller."""
    _, dataset, metadata = run.provenance
    import numpy as np
    import torch

    from benchmark.contracts import ObservationActionSpec, camera_image
    from policy.config import cameras
    from benchmark.learning.evaluation_env import PolicyEvaluationSession, policy_task
    from benchmark.catalog import is_tabletop_task
    from benchmark.tasks.success import TaskSuccessMonitor, success_contract
    from benchmark.io.video import VISION_VIDEO_CRF, VISION_VIDEO_PRESET, rgb_video
    from benchmark.backends.gsmini.execution import step_env
    from benchmark.backends.gsmini.markers import marker_images

    spec = ObservationActionSpec.from_dict(metadata["contract"])
    from benchmark.scene.perturbations import EvalPerturbationSpec, perturb_task
    look = getattr(args, "perturbation", EvalPerturbationSpec())
    task = perturb_task(policy_task(args.suite, args.task_id), args.suite, look)
    video_mode = getattr(args, "video_mode", "all")
    policy_cameras = cameras(args.policy_type)
    camera_names = spec.camera_names if video_mode == "all" else policy_cameras
    if video_mode == "all" or args.policy_type == "tacmind0":
        # TacMind-0 derives Deformation inputs even when preview video is disabled.
        camera_names = (*camera_names, "gsmini_left", "gsmini_right")
    camera_shapes = {camera: spec.camera_shape(camera) for camera in camera_names}
    video_shapes = {**camera_shapes, **{
        f"{view}_{side}": camera_shapes[f"gsmini_{side}"]
        for side in ("left", "right") for view in ("marker", "deformation")}} if video_mode == "all" else {}
    runner, policy_report = run.policy(args, spec, metadata)
    report.update(policy_report, dataset=str(dataset), seeds=seeds,
                  success_contract=success_contract(task, suite=args.suite, backend="gsmini"),
                  video_mode=video_mode, max_simulation_seconds=args.max_steps / spec.fps,
                  perturbation=look.to_dict())
    native_images = args.policy_type in EXTERNAL_TYPES
    session = PolicyEvaluationSession(app, task, args.suite, spec, metadata.get("recording_contract", {}),
                                      perturbation=look)
    with session:
        for seed in seeds:
            with session.episode(seed) as (env, action):
                arm = env.action_manager.get_term("arm_action")
                torch.manual_seed(seed)
                runner.reset()
                monitor = TaskSuccessMonitor(env, task, suite=args.suite, backend="gsmini")
                initialization_steps = (env.start_policy_episode(monitor, args.max_steps)
                                        if hasattr(env, "start_policy_episode") else None)
                row = {"seed": seed, "success": False, "steps": 0, "status": "running", "videos": {},
                       "perturbation": env.evaluation_perturbation,
                       "initialization_steps": initialization_steps}
                report["current_episode"] = row
                write_json(args.output_dir / "results.json", report)
                trace, started = [], monotonic()
                try:
                    with ExitStack() as videos:
                        writers = {}
                        for camera, shape in video_shapes.items():
                            path = args.output_dir / "videos" / f"seed{seed:05d}_{camera}.mp4"
                            row["videos"][camera] = str(path)
                            height, width = shape[:2]
                            writers[camera] = videos.enter_context(rgb_video(
                                path, width=width, height=height, fps=spec.fps,
                                crf=VISION_VIDEO_CRF, preset=VISION_VIDEO_PRESET,
                                encoder_threads=2, buffered=True))
                        for step in range(args.max_steps):
                            needs_observation = getattr(runner, "needs_observation", True)
                            state = env.scene["robot"].data.joint_pos[0].to(dtype=torch.float32).clone()
                            observation = {"observation.state": state}
                            for camera, shape in camera_shapes.items():
                                if not needs_observation and camera not in writers:
                                    continue
                                image = camera_image(env, camera)
                                pixels = image.cpu().numpy() if camera in writers or native_images else None
                                if camera in writers:
                                    writers[camera](pixels)
                                if camera in ("gsmini_left", "gsmini_right") and (
                                        camera in writers or needs_observation and args.policy_type == "tacmind0"):
                                    side = camera.removeprefix("gsmini_")
                                    motion = env.scene[camera].data.output["marker_motion"][0]
                                    marker, deformation = marker_images(pixels, motion.detach().cpu().numpy())
                                    if camera in writers:
                                        writers[f"marker_{side}"](marker)
                                        writers[f"deformation_{side}"](deformation)
                                    if needs_observation and args.policy_type == "tacmind0":
                                        observation[f"observation.images.deformation_{side}"] = torch.from_numpy(deformation)
                                if needs_observation and camera in policy_cameras:
                                    if native_images:
                                        # Native adapters consume raw CPU bytes.
                                        image = torch.from_numpy(pixels)
                                    image = image.permute(2, 0, 1)
                                    observation[f"observation.images.{camera}"] = (
                                        image if native_images else image.float().div_(255.0))
                            target = runner.select_action(observation).to(env.device, dtype=action.dtype)
                            action[:, :7] = arm.position_actions(target[:, :7])
                            action[:, -1] = target[:, -1]
                            if getattr(args, "trace", False):
                                obj = (env.diagnostic_object.data if args.suite == "tac_extra" and not is_tabletop_task(task)
                                       else env.scene["object"].data)
                                tcp = env.scene["ee_frame"].data
                                trace.append(torch.cat((state, target[0], action[0], obj.root_pos_w[0],
                                                        obj.root_quat_w[0], tcp.target_pos_w[0, 0],
                                                        tcp.target_quat_w[0, 0])).cpu().numpy())
                            terminated, truncated = step_env(env, app, action, return_termination=True)
                            row["steps"] = step + 1
                            row.update(terminated=bool(terminated.any()), truncated=bool(truncated.any()))
                            final_goal = getattr(env, "policy_terminal_goal_status", None)
                            row["goal_status"] = final_goal if final_goal is not None else monitor.update()
                            if row["goal_status"]["success"]:
                                row.update(success=True, reason="shared task success monitor")
                                break
                            if row["terminated"] or row["truncated"]:
                                row["reason"] = "environment termination" if row["terminated"] else "policy time limit"
                                break
                            if (step + 1) % 50 == 0 and step + 1 < args.max_steps:
                                write_json(args.output_dir / "results.json", report)
                                print(f"[EVAL] seed={seed} step={step + 1}/{args.max_steps}", flush=True)
                        else:
                            row["reason"] = f"{args.max_steps}-step evaluation limit"
                    row["status"] = "completed"
                except BaseException as error:
                    row.update(status="error", error=f"{type(error).__name__}: {error}")
                    raise
                finally:
                    row["elapsed_seconds"] = monotonic() - started
                    row["simulation_seconds"] = row["steps"] / spec.fps
                    if trace:
                        path = args.output_dir / f"seed{seed:05d}_trace.npz"
                        np.savez_compressed(path, pre_action=np.stack(trace),
                                            columns=f"joint_pos:9,policy_target:{action.shape[-1]},environment_action:8,"
                                                    "object_pose:7,tcp_pose:7")
                        row["trace"] = str(path)
                report["episodes"].append(row)
                report.pop("current_episode")
                report.update(episodes_completed=len(report["episodes"]),
                              successes=report["successes"] + int(row["success"]))
                report["completed_episode_success_rate"] = report["successes"] / report["episodes_completed"]
                write_json(args.output_dir / "results.json", report)
                print(f"[RESULT] seed={seed} success={row['success']} steps={row['steps']} "
                      f"total={report['successes']}/{report['episodes_completed']}", flush=True)


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    show_advanced = "--help-advanced" in arguments
    parser = argparse.ArgumentParser(
        prog="tacmind learning evaluate", allow_abbrev=False,
        description="Evaluate a checkpoint on held-out seeds using its saved task and execution settings.")
    parser.add_argument("--help-advanced", action="help", help="Show diagnostics and DP temporal ensembling")
    def advanced_help(text):
        return text if show_advanced else argparse.SUPPRESS

    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.set_defaults(suite=None, task_id=None, policy_type=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--video-mode", choices=("none", "all"), default="all",
                        help="Save all diagnostic videos per episode (default: all); use none to save scores only")
    parser.add_argument("--seed-start", type=int, default=10000,
                        help="First held-out evaluation seed (default: 10000)")
    parser.add_argument("--training-seeds", type=int, nargs="+",
                        help=advanced_help("Recorded seeds for diagnostics; replaces the episode count"))
    parser.add_argument("--temporal-agg", action="store_true", default=None,
                        help=advanced_help("DP: replan each step and blend overlapping chunks"))
    parser.add_argument("--temporal-coeff", type=float,
                        help=advanced_help("DP temporal ensemble coefficient (default: 0.01); requires --temporal-agg"))
    parser.add_argument("--policy-python", help="Isolated model Python interpreter (overrides saved path)")
    parser.add_argument("--trace", action="store_true", help=advanced_help("Save pre-action diagnostic traces"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Show the evaluation settings without loading model weights or Isaac")
    from benchmark.scene.perturbations import PERTURBATION_KINDS, expand_eval_perturbations
    parser.add_argument('--perturb', nargs='*', choices=PERTURBATION_KINDS,
                        help='Precision, Recog and Dynamic only; omitted disables perturbations; '
                             'no kinds runs all 12 fixed presets independently')
    if show_advanced or "--help" in arguments or "-h" in arguments:
        parser.epilog = "GSmini also accepts Isaac Lab AppLauncher flags such as --headless and --device."
        parser.print_help()
        return
    # Read evaluation options before loading a simulator.
    # Backend launcher arguments are parsed by the selected backend below.
    selected, simulator_arguments = parser.parse_known_args(arguments)
    try:
        seeds = evaluation_seeds(selected)
        selected.checkpoint = selected.checkpoint.expanduser().resolve()
        provenance = checkpoint_metadata(selected.checkpoint)
        training, _, metadata = provenance
        saved = resolve_checkpoint_options(selected, training, metadata)
        if selected.suite == "tac_extra" and selected.perturb is not None:
            raise ValueError("ExtraSuite does not support evaluation perturbations; omit --perturb")
        looks = expand_eval_perturbations(selected.perturb)
    except ValueError as error:
        parser.error(str(error))
    if selected.dry_run:
        print(json.dumps(dict(
            checkpoint=str(selected.checkpoint), policy_type=selected.policy_type,
            suite=selected.suite, task_id=selected.task_id,
            seeds=seeds, max_steps=selected.max_steps,
            video_mode=selected.video_mode,
            output_dir=str(selected.output_dir.expanduser().resolve()),
            saved_execution={key: saved[key] for key in (
                "chunk_size", "horizon", "n_obs_steps", "n_action_steps", "temporal_ensemble_coeff") if key in saved},
            inference_options={key: getattr(selected, key) for key in (
                "temporal_agg", "temporal_coeff", "policy_python")},
            perturbations={name: look.to_dict() for name, look in looks},
            simulator_arguments=simulator_arguments,
        ), indent=2))
        return
    import warp  # noqa: F401 -- load conda Warp before Kit's bundled version
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(arguments)
    # Keep checkpoint-derived options from the first pass; add launcher options.
    vars(args).update(vars(selected))
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        raise FileExistsError(f"Evaluation requires an empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.enable_cameras = True
    from benchmark.backends.gsmini.execution import simulation_app

    from benchmark.learning.perturbations import run_evaluation_looks
    with simulation_app(AppLauncher(args)) as app, EvaluationRun(args, provenance=provenance) as run:
        run_evaluation_looks(app, args, looks, run)


if __name__ == "__main__":
    main()
