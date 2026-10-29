"""Collect benchmark jobs concurrently in independent simulation processes."""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from benchmark.expert.capabilities import BACKENDS, ENVIRONMENT_TASK_IDS, SUITES, process_reuse_supported, supported_task_ids
from benchmark.backends.gsmini.recording import EPISODE_COMPRESSION
from benchmark.io import write_json


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TIMEOUT_SECONDS = {"gsmini": 600.}
EXTRA_TIMEOUT_SECONDS = 3600.
EXTRA_SCENE_TIMEOUT_SECONDS = 10800.
RESTART_GRACE_SECONDS = 30.


def write_collection_report(path, report):
    """Keep saved recording and log paths relative to their summary."""
    path = Path(path)
    rows = [{key: os.path.relpath(value, path.parent)
             if key in ("output", "log", "failed_output") else value
             for key, value in row.items()} for row in report["jobs"]]
    write_json(path, {**report, "jobs": rows})


def default_timeout(jobs, backend="gsmini"):
    """Allow long rendered Extra scene episodes without changing simulation limits."""
    timeout = DEFAULT_TIMEOUT_SECONDS[backend]
    for job in jobs:
        if not isinstance(job, dict) or job.get("suite") != "extra":
            continue
        if job.get("task_id") not in ENVIRONMENT_TASK_IDS["extra"][backend]:
            continue
        task_timeout = (EXTRA_SCENE_TIMEOUT_SECONDS if job.get("task_id", -1) >= 21
                        else EXTRA_TIMEOUT_SECONDS)
        timeout = max(timeout, task_timeout)
    return timeout


def _command(job, output, *, backend="gsmini"):
    command = [sys.executable, "-u", "-m", "benchmark.expert.collect",
               "--suite", job["suite"], "--backend", backend,
               "--task-id", str(job["task_id"]), "--seed", str(job["seed"]),
               "--headless", "--device", "cuda:0", "--output", str(output)]
    return command


def _job_stem(job):
    return f"{job['suite']}{job['task_id']}_seed{job['seed']:03d}"


def _fatal_simulator_error(content):
    for message in (b"Cuda context manager error, simulation will be stopped",
                    b"CUDA error: an illegal memory access was encountered",
                    b"CUDA error: misaligned address"):
        if message in content:
            return message.decode()
    for line in content.splitlines():
        prefix = b"PhysX error: GPU "
        if prefix in line and b"fail to launch" in line:
            return (prefix + line.partition(prefix)[2]).decode(errors="replace")
    return None


def _restart_requested(worker):
    return (worker.get("stop_reason") == "environment_restart"
            and "active_seed" in worker and worker["active_seed"] is None
            and bool(worker.get("jobs")) and worker["jobs"][-1].get("status") == "failed")


def _save_worker_log(output_dir, row, log):
    log.flush()
    offset = getattr(log, "_saved_bytes", 0)
    end = os.fstat(log.fileno()).st_size
    if offset == end and row["status"] == "running":
        return getattr(log, "_fatal_error", None)
    with (output_dir / "collection.log").open("ab") as combined:
        combined.write(f"\n=== {_job_stem(row)}: {row['status']} ===\n".encode())
        # pread leaves the shared worker write offset untouched. Bound memory
        # and only copy bytes not already published by an earlier poll.
        while offset < end:
            content = os.pread(log.fileno(), min(64 * 1024, end - offset), offset)
            if not content:
                break
            combined.write(content)
            scan = getattr(log, "_scan_tail", b"") + content
            log._fatal_error = getattr(log, "_fatal_error", None) or _fatal_simulator_error(scan)
            log._scan_tail = scan[-128:]
            offset += len(content)
        combined.write(b"\n")
    log._saved_bytes = offset
    return getattr(log, "_fatal_error", None)


def _recording_frames(output):
    """Read the saved episode length for the collection summary."""
    import h5py

    with h5py.File(output, "r") as dataset:
        return len(dataset["data/demo_0/actions"])


def _stop(process):
    # Kit and any helper processes belong to the worker's own process group.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # A helper can ignore SIGTERM even if Kit itself has already exited.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _reject_output(row):
    output = Path(row["output"])
    if output.is_file():
        failed = output.with_suffix(".hdf5.failed")
        output.rename(failed)
        row["failed_output"] = str(failed)


def collect_parallel(jobs, output_dir: Path, *, workers: int = 2, timeout: float | None = None,
                     camera_resolution=None,
                     episodes_per_worker: int | None = None, backend: str = "gsmini",
                     success_target: int | None = None, progress=None, vision_crf: int = 0):
    jobs = [dict(suite=job["suite"], task_id=job["task_id"], seed=job.get("seed", 0)) for job in jobs]
    if len({_job_stem(job) for job in jobs}) != len(jobs):
        raise ValueError("Duplicate jobs would write to the same episode output")
    timeout = default_timeout(jobs, backend) if timeout is None else timeout
    if type(workers) is not int or workers < 1 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Workers must be positive and timeout must be finite and positive")
    reusable = all(
        process_reuse_supported(job["suite"], job["task_id"]) for job in jobs)
    if episodes_per_worker is None:
        episodes_per_worker = max(len(jobs), 1) if reusable else 1
    if type(episodes_per_worker) is not int or episodes_per_worker < 1:
        raise ValueError("Episodes per worker must be a positive integer")
    if success_target is not None and (workers != 1
            or type(success_target) is not int or not 1 <= success_target <= len(jobs)
            or episodes_per_worker < len(jobs) or len(jobs) < 2
            or len({(job["suite"], job["task_id"]) for job in jobs}) != 1):
        raise ValueError("A success target requires one worker with all seeds of one task")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True)  # An exclusive run directory prevents overwrites and collisions.
    started = time.monotonic()
    report = {"backend": backend, "workers": workers, "timeout_seconds": timeout,
              "compression": EPISODE_COMPRESSION, "vision_crf": vision_crf,
              "episodes_per_worker": episodes_per_worker, "jobs": []}
    report["timeout_scope"] = "episode"
    if success_target is not None:
        report["success_target"] = success_target
    for job in jobs:
        stem = _job_stem(job)
        report["jobs"].append({**job, "status": "queued",
                               "output": str(output_dir / f"{stem}.hdf5"),
                               "log": str(output_dir / "collection.log")})

    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        write_collection_report(output_dir / "summary.json", report)

    def finish(indices, process, log, elapsed, terminal, fatal_error=None):
        records, error = {}, f"Worker exited with code {process.returncode}; see collection.log"
        active_seed = None
        restart_requested = False
        if terminal == "timed_out":
            error = f"Worker exceeded {timeout:g} seconds"
        elif terminal == "cancelled":
            error = "Worker cancelled"
        if fatal_error:
            error = f"Unrecoverable simulator error: {fatal_error}"
        grouped = len(indices) > 1
        if grouped:
            path = Path(report["jobs"][indices[0]]["output"]).with_suffix(".worker.json")
            try:
                worker = json.loads(path.read_text())
                records = {entry["seed"]: entry for entry in worker["jobs"]}
                active_seed = worker.get("active_seed")
                restart_requested = _restart_requested(worker)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                error = f"Cannot read completed worker results: {exc}"
        log.flush()
        log.seek(0)
        fatal_errors, current_seed = {}, None
        for line in log:
            if line.startswith(b"[EPISODE_START] "):
                try:
                    current_seed = int(line.split()[1])
                except (ValueError, IndexError):
                    current_seed = None
            if fatal := _fatal_simulator_error(line):
                fatal_errors[current_seed] = fatal
        for index in indices:
            row = report["jobs"][index]
            if success_target is not None and row["seed"] not in records:
                # Retain an interrupted attempt, but do not consume unstarted
                # seeds after a quota, failure limit, crash or shutdown.
                if (row["seed"] not in (active_seed, current_seed)
                        and (records or index != indices[0])):
                    row["status"] = "skipped"
                    continue
            if (grouped and row["seed"] not in records
                    and (process.returncode == 0 or restart_requested or fatal_error)
                    and row["seed"] not in (active_seed, current_seed)
                    and (records or index != indices[0])
                    and success_target is None
                    and terminal == "failed"
                    and (fatal_error or any(r["status"] == "failed" for r in records.values()))):
                row["status"] = "queued"
                pending.append(index)
                continue
            entry = records.get(row["seed"], {"status": terminal, "error": error}) if grouped else (
                {"status": "success"} if process.returncode == 0 and terminal == "failed" and not fatal_error else
                {"status": terminal, "error": error})
            row.update(status=entry["status"], returncode=process.returncode,
                       elapsed_seconds=entry.get("elapsed_seconds", elapsed))
            if fatal := fatal_errors.get(row["seed"], fatal_errors.get(None)):
                row.update(status="failed", error=f"Unrecoverable simulator error: {fatal}")
                entry = {**entry, "error": row["error"]}
            if row["status"] == "success":
                try:
                    row["frames"] = _recording_frames(Path(row["output"]))
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    row.update(status="failed", error=str(exc))
            else:
                row["error"] = entry.get("error", error)
            if row["status"] != "success":
                _reject_output(row)
            print(f"[DONE] {row['suite']} {row['task_id']} seed={row['seed']}: {row['status']}", flush=True)
        _save_worker_log(output_dir, report["jobs"][indices[0]], log)

    active, pending = {}, deque(range(len(jobs)))
    published, last_progress = {}, {}
    environment = {**os.environ, "HEADLESS": "1", "LIVESTREAM": "0"}
    try:
        save()
        while pending or active:
            while pending and len(active) < workers:
                size = min(episodes_per_worker, math.ceil(len(pending) / (workers - len(active))))
                indices = [pending.popleft()]
                first = jobs[indices[0]]
                while (len(indices) < size and pending
                       and (jobs[pending[0]]["suite"], jobs[pending[0]]["task_id"])
                       == (first["suite"], first["task_id"])):
                    indices.append(pending.popleft())
                index = indices[0]
                row = report["jobs"][index]
                log = tempfile.TemporaryFile(mode="w+b")
                try:
                    command = _command(jobs[index], Path(row["output"]))
                    if len(indices) > 1:
                        command += ["--additional-seeds", *(str(jobs[i]["seed"]) for i in indices[1:])]
                    if success_target is not None:
                        command += ["--success-target", str(success_target)]
                    if camera_resolution is not None:
                        command += ["--camera-resolution", *map(str, camera_resolution)]
                    if vision_crf:
                        command += ["--vision-crf", str(vision_crf)]
                    process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                except Exception as error:
                    for i in indices:
                        report["jobs"][i].update(status="failed", error=str(error))
                        _reject_output(report["jobs"][i])
                    _save_worker_log(output_dir, row, log)
                    log.close()
                    save()
                    continue
                for i in indices:
                    report["jobs"][i].update(status="running", pid=process.pid, command=command,
                                            started_seconds=time.monotonic() - started)
                active[index] = (process, log, indices)
                last_progress[index] = time.monotonic()
                print(f"[START] {row['suite']} {row['task_id']} seed={row['seed']}", flush=True)
                save()
            for index, (process, log, indices) in list(active.items()):
                row = report["jobs"][index]
                elapsed = time.monotonic() - started - row["started_seconds"]
                code = process.poll()
                restart_requested = False
                if len(indices) > 1:
                    path = Path(row["output"]).with_suffix(".worker.json")
                    try:
                        worker = json.loads(path.read_text())
                        entries = worker["jobs"]
                        restart_requested = _restart_requested(worker)
                    except (OSError, ValueError, KeyError):
                        entries = []
                    if len(entries) > published.get(index, 0):
                        published[index] = len(entries)
                        last_progress[index] = time.monotonic()
                        if progress is not None:
                            by_seed = {entry["seed"]: entry for entry in entries}
                            progress([{**report["jobs"][i], **by_seed[jobs[i]["seed"]]}
                                      for i in indices if jobs[i]["seed"] in by_seed])
                waiting = time.monotonic() - last_progress[index] if len(indices) > 1 else elapsed
                # The failed attempt is already durable and no episode is in
                # flight. Give Kit a short teardown grace, not another episode.
                restart_due = restart_requested and waiting > RESTART_GRACE_SECONDS
                fatal_error = _save_worker_log(output_dir, row, log)
                if code is None and waiting <= timeout and not fatal_error and not restart_due:
                    continue
                if code is None:
                    if restart_due:
                        print(f"[RESTART] {_job_stem(row)}: failed-session teardown exceeded "
                              f"{RESTART_GRACE_SECONDS:g} seconds", flush=True)
                    _stop(process)
                finish(indices, process, log, elapsed,
                       "timed_out" if code is None and not fatal_error and not restart_due else "failed", fatal_error)
                log.close()
                del active[index]
                save()
            if active:
                time.sleep(0.1)
    finally:
        for index, (process, log, indices) in active.items():
            _stop(process)
            elapsed = time.monotonic() - started - report["jobs"][index]["started_seconds"]
            finish(indices, process, log, elapsed, "cancelled")
            log.close()
        for row in report["jobs"]:
            if row["status"] == "queued":
                row["status"] = "cancelled"
        report["jobs"] = [row for row in report["jobs"] if row["status"] != "skipped"]
        save()
        if progress is not None:
            progress(report["jobs"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BACKENDS, default="gsmini", help=argparse.SUPPRESS)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--suite", choices=SUITES)
    source.add_argument("--jobs", type=Path, help="JSON list of {suite, task_id, seed} jobs")
    parser.add_argument("--task-ids", type=int, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+", help="Episode seeds (default: 0)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Concurrent workers (default: 2)")
    parser.add_argument("--episodes-per-worker", type=int,
                        help="Episodes per persistent worker (default: group supported same-task jobs).")
    parser.add_argument("--camera-resolution", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--vision-crf", type=int, default=0,
                        help="Visual RGB compression: 0=lossless (default), 12=compact; tactile stays lossless.")
    parser.add_argument("--timeout", type=float, default=None,
                        help="Wall-clock timeout per episode, in seconds "
                             "(default: 10800 for GSmini Extra scene, 3600 for GSmini Extra, 600 otherwise)")
    parser.add_argument("--output-dir", type=Path, required=True, help="New, exclusive run directory")
    args = parser.parse_args()
    if args.jobs is not None and (args.task_ids is not None or args.seeds is not None):
        parser.error("--task-ids and --seeds require --suite")
    try:
        jobs = (json.loads(args.jobs.read_text()) if args.jobs else
                [dict(suite=args.suite, task_id=task, seed=seed)
                 for task in (args.task_ids if args.task_ids is not None else supported_task_ids(args.suite, args.backend))
                 for seed in (args.seeds if args.seeds is not None else [0])])
        def interrupt(signum, frame):
            raise KeyboardInterrupt
        previous = signal.signal(signal.SIGTERM, interrupt)
        try:
            workers = args.workers if args.workers is not None else 2
            timeout = args.timeout if args.timeout is not None else default_timeout(jobs, args.backend)
            report = collect_parallel(jobs, args.output_dir, workers=workers, timeout=timeout,
                                      camera_resolution=args.camera_resolution,
                                      vision_crf=args.vision_crf,
                                      episodes_per_worker=args.episodes_per_worker, backend=args.backend)
        finally:
            signal.signal(signal.SIGTERM, previous)
    except KeyboardInterrupt:
        print("[CANCELLED] Workers stopped; inspect summary.json and collection.log", file=sys.stderr)
        raise SystemExit(130)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    raise SystemExit(0 if all(row["status"] == "success" for row in report["jobs"]) else 1)


if __name__ == "__main__":
    main()
