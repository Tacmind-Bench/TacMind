"""Collect Tac expert episodes across suites and tactile backends."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from importlib import import_module
import json
import sys
from pathlib import Path
from time import perf_counter
import traceback

from benchmark.expert.capabilities import BACKENDS, REUSABLE_GSMINI_TASK_IDS, SUITES
from benchmark.io import write_json


def _tactile_output_path(output):
    """Reject an existing export location before starting collection."""
    tactile = output.with_suffix('')/'tactile_export'
    if tactile.exists():
        raise FileExistsError(f"Tactile export already exists: {tactile}")
    return tactile


def collect(app, suite: str, backend: str, seed: int, output: Path, task_id: int, *, session=None,
            vision_crf: int = 0):
    """Collect a successful episode; GSmini also publishes all tactile views and force arrays."""
    from benchmark.backends.gsmini.recording import vision_compression

    tactile = _tactile_output_path(output)

    run = import_module(f"benchmark.tasks.{suite}.gsmini").run
    started = perf_counter()
    options = {}
    if session is not None:
        options["session"] = session
    with vision_compression(vision_crf):
        result = run(app, seed, output, task_id, **options)
    print(f"[TIMING] {suite}/{backend} expert_collection_s={perf_counter() - started:.3f}", flush=True)
    from benchmark.backends.gsmini.export import export_episode

    export_episode(output, tactile)
    return result


def _collect_sequence(app, args):
    """Reuse collection sessions; let the supervisor restart after session failure."""
    from benchmark.expert.tabletop import TubeCollectionSession
    from benchmark.backends.gsmini.environment import GSminiCollectionSession

    report = {"jobs": [], "stop_reason": "attempt_limit"}
    report_path = args.output.with_suffix(".worker.json")
    target = getattr(args, "success_target", None)
    successes = consecutive = 0
    episode_session = None
    with ExitStack() as resources:
        for seed in (args.seed, *args.additional_seeds):
            output = (args.output if seed == args.seed else
                      args.output.parent / f"{args.suite}{args.task_id}_seed{seed:03d}.hdf5")
            row = {"seed": seed, "output": str(output), "status": "running"}
            report["active_seed"] = seed
            write_json(report_path, report)
            print(f"[EPISODE_START] {seed}", flush=True)
            started = perf_counter()
            try:
                if (episode_session is None
                        and args.task_id in REUSABLE_GSMINI_TASK_IDS.get(args.suite, ())):
                    session = (TubeCollectionSession() if (args.suite, args.task_id) == ("precision", 0)
                               else GSminiCollectionSession(args.suite, args.task_id))
                    episode_session = resources.enter_context(session)
                collect(app, args.suite, args.backend, seed, output, args.task_id, session=episode_session,
                        vision_crf=getattr(args, "vision_crf", 0))
                if target is not None:
                    from benchmark.expert.collect_parallel import _recording_frames

                    row["frames"] = _recording_frames(output)
                row["status"] = "success"
                successes += 1
                consecutive = 0
            except Exception:
                row.update(status="failed", error=traceback.format_exc())
                print(row["error"], flush=True)
                # Rebuilding a failed session in the same Kit process can
                # stall. Persist this attempt before teardown, then restart Kit.
                if episode_session is not None:
                    report["stop_reason"] = "environment_restart"
                consecutive += 1
            row["elapsed_seconds"] = perf_counter() - started
            report["jobs"].append(row)
            report["active_seed"] = None
            if target is not None and successes >= target:
                report["stop_reason"] = "success_target"
            elif consecutive >= 5:
                report["stop_reason"] = "consecutive_failures"
            write_json(report_path, report)
            print(f"[EPISODE_RESULT] {json.dumps(row)}", flush=True)
            if report["stop_reason"] != "attempt_limit":
                break
    return report


def _collect_with_launcher(launcher, args):
    """Run only the tactile backends that use the Isaac environment."""
    from benchmark.backends.gsmini.execution import simulation_app

    with simulation_app(launcher, require_completion=True) as app:
        if getattr(args, "additional_seeds", None):
            result = _collect_sequence(app, args)
        else:
            result = collect(
                app, args.suite, args.backend, args.seed, args.output, args.task_id,
                vision_crf=getattr(args, "vision_crf", 0),
            )
        shutdown_started = perf_counter()
    print(f"[TIMING] app_shutdown_s={perf_counter() - shutdown_started:.3f}", flush=True)
    return result


def main(argv=None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--backend", choices=BACKENDS, default="gsmini", help=argparse.SUPPRESS)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--additional-seeds", type=int, nargs="+",
                        help="Collect more GSmini episodes without restarting Isaac Sim.")
    parser.add_argument("--success-target", type=int,
                        help="Stop a GSmini sequence after this many successful episodes.")
    parser.add_argument("--camera-resolution", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"),
                        help="Override both RGB camera resolutions for this recording.")
    parser.add_argument("--vision-crf", type=int, default=0,
                        help="Visual RGB compression: 0=lossless (default), 12=compact; range 0–51. Tactile stays lossless.")
    parser.add_argument("--viewport-width", type=int, default=960, dest="width",
                        help="GUI viewport width in pixels (does not change scene camera data).")
    parser.add_argument("--viewport-height", type=int, default=540, dest="height",
                        help="GUI viewport height in pixels (does not change scene camera data).")
    parser.add_argument("--output", type=Path, required=True)
    if "--help" in arguments or "-h" in arguments:
        parser.epilog = ("Isaac launcher options include --headless, --device cuda:0, "
                         "--livestream and --kit_args.")
        parser.print_help()
        return
    parser.parse_known_args(arguments)
    from benchmark.backends.gsmini.recording import _prepare_output_path
    import warp  # noqa: F401 -- load installed Warp before Kit's bundled version
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(arguments)
    seeds = [args.seed, *(args.additional_seeds or ())]
    if len(set(seeds)) != len(seeds):
        parser.error("Collection seeds must be unique to keep episode outputs distinct")
    if args.camera_resolution is not None:
        from benchmark.scene.cameras import CAMERAS

        for spec in CAMERAS.values():
            spec["resolution"] = tuple(args.camera_resolution)
    outputs = [args.output] + [
        args.output.parent / f"{args.suite}{args.task_id}_seed{seed:03d}.hdf5"
        for seed in args.additional_seeds or ()
    ]
    try:
        for output in outputs:
            _prepare_output_path(output)
        for output in outputs:
            _tactile_output_path(output)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))

    args.enable_cameras = True
    started = perf_counter()
    launcher = AppLauncher(args)
    print(f"[TIMING] app_startup_s={perf_counter() - started:.3f}", flush=True)
    _collect_with_launcher(launcher, args)


if __name__ == "__main__":
    main()
