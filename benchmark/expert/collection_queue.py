"""Keep collection GPUs supplied with tasks and archive completed runs to SSD."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from benchmark.catalog import ROOT
from benchmark.io import write_json
from benchmark.expert.capabilities import process_reuse_supported
from benchmark.expert.collect_parallel import (
    _recording_frames, collect_parallel, default_timeout, write_collection_report,
)

FAILED_STATUSES = ("failed", "timed_out", "cancelled")
DEFAULT_MAX_ATTEMPTS = 500
SEED_LIMIT = 2**32


def selected_gpu(gpu):
    """Select a physical device without widening the caller's visible devices."""
    value = str(gpu)
    if not (re.fullmatch(r"0|[1-9][0-9]*", value)
            or re.fullmatch(r"GPU-[0-9a-fA-F-]+", value)):
        raise ValueError("GPU must be a nonnegative physical index or a GPU UUID/prefix")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return value
    allowed = [entry.strip() for entry in visible.split(",")]
    if any(not (re.fullmatch(r"0|[1-9][0-9]*", entry)
                or re.fullmatch(r"GPU-[0-9a-fA-F-]+", entry)) for entry in allowed):
        raise ValueError(f"No selectable GPU in CUDA_VISIBLE_DEVICES={visible!r}")
    if value in allowed:
        return value
    if value.startswith("GPU-") or any(entry.startswith("GPU-") for entry in allowed):
        inventory = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True, timeout=10)
        devices = [tuple(part.strip() for part in line.split(",")) for line in inventory.splitlines()]
        permitted = set()
        for entry in allowed:
            matches = [uuid for index, uuid in devices if entry == index or uuid.startswith(entry)]
            if len(matches) != 1:
                raise ValueError(f"Unknown or ambiguous visible GPU: {entry}")
            permitted.add(matches[0])
        matches = [uuid for index, uuid in devices if value == index or uuid.startswith(value)]
        if len(matches) == 1 and matches[0] in permitted:
            return matches[0]
    raise ValueError(f"GPU {value} is outside CUDA_VISIBLE_DEVICES={visible!r}")


def _queue_devices(slots):
    devices = {gpu: selected_gpu(gpu) for gpu in slots}
    if len(devices) > 1 and any(value.startswith("GPU-") for value in devices.values()):
        inventory = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True, timeout=10)
        available = [tuple(part.strip() for part in line.split(",")) for line in inventory.splitlines()]
        for gpu, value in devices.items():
            matches = [uuid for index, uuid in available if value == index or uuid.startswith(value)]
            if len(matches) != 1:
                raise ValueError(f"Unknown or ambiguous queue GPU: {gpu}")
            devices[gpu] = matches[0]
    if len(set(devices.values())) != len(devices):
        raise ValueError("Queue GPU slots must select distinct physical devices")
    return devices


def _physical_rejection(row):
    """Recognize completed physics attempts, never infrastructure or data errors."""
    if row.get("status") != "failed":
        return False
    message = row.get("error", "").strip().splitlines()
    if not message:
        return False
    terminal = message[-1]
    return terminal.startswith((
        "benchmark.backends.gsmini.environment.PlanningFailed:",
        "benchmark.tasks.success.GoalNotReached:",
        "RuntimeError: Grasp lost:",
        "RuntimeError: Source payload is not seated in its holder:",
        "RuntimeError: Moving key did not establish stable bilateral contact within 10s:",
    )) or (terminal.startswith("RuntimeError: Moving key ")
           and " did not physically converge within 8s:" in terminal)


def process_alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except FileNotFoundError:
        return False


def _resume_report(path, suite, task_id, backend, *, task_summary=False):
    """Check report identity before any rows can count toward a resumed quota."""
    report = json.loads(path.read_text())
    if (report["backend"] != backend
            or (task_summary and (report["suite"] != f"tac_{suite}" or report["task_id"] != task_id))):
        raise ValueError(f"Cannot resume collection with a different backend or task: {path}")
    seen = set()
    for row in report["jobs"]:
        if row["suite"] != suite or row["task_id"] != task_id:
            raise ValueError(f"Cannot resume collection with a different task: {path}")
        if row["seed"] in seen:
            raise ValueError(f"Duplicate seed {row['seed']} in collection report: {path}")
        seen.add(row["seed"])
        for key in ("output", "log", "failed_output"):
            if key in row:
                row[key] = str((path.parent / row[key]).resolve())
    return report


def collect_task(suite, task_id, gpu, output, *, target=100, backend="gsmini", vision_crf=0,
                 seed_start=0, max_attempts=DEFAULT_MAX_ATTEMPTS):
    """Collect until the success target or attempt budget is reached."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".task.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _collect_task(suite, task_id, gpu, output, target=target,
                             backend=backend, vision_crf=vision_crf,
                             seed_start=seed_start, max_attempts=max_attempts)


def _collect_task(suite, task_id, gpu, output, *, target, backend, vision_crf, seed_start, max_attempts):
    """Resume and replenish a task while its output directory is locked."""
    summary = output / "summary.json"
    if summary.exists():
        report = _resume_report(summary, suite, task_id, backend, task_summary=True)
    else:
        report = {"backend": backend, "suite": f"tac_{suite}", "task_id": task_id,
                  "jobs": [], "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    rows = {row["seed"]: row for row in report["jobs"]}
    worker_seeds = set()
    for path in sorted(output.glob("seed*/summary.json")):
        worker = _resume_report(path, suite, task_id, backend)
        for row in worker["jobs"]:
            if row["status"] in ("running", "queued"):
                raise RuntimeError(f"Unreconciled worker: {path}")
            if row["seed"] in worker_seeds:
                raise ValueError(f"Duplicate seed {row['seed']} across worker reports: {path}")
            worker_seeds.add(row["seed"])
            rows[row["seed"]] = row
    if any(row["status"] in ("running", "queued") for row in rows.values()):
        raise RuntimeError(f"Unreconciled worker: {summary}")
    report["jobs"] = [rows[seed] for seed in sorted(rows)]
    for row in report["jobs"]:
        if row["status"] == "success":
            try:
                row["frames"] = _recording_frames(Path(row["output"]))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"Cannot resume invalid success at seed {row['seed']}: {exc}") from exc
    report.update(status="running", gpu=gpu, pid=os.getpid(), target=target,
                  seed_start=seed_start, max_attempts=max_attempts,
                  camera_resolution=[448, 448], vision_crf=vision_crf)
    report.pop("reason", None)
    report.pop("error", None)
    continuous = process_reuse_supported(suite, task_id)
    size = min(DEFAULT_MAX_ATTEMPTS, max_attempts) if continuous else 1
    report["episodes_per_worker"] = size

    def save():
        report["success_count"] = sum(row["status"] == "success" for row in report["jobs"])
        report["failed_count"] = sum(row["status"] in FAILED_STATUSES for row in report["jobs"])
        write_collection_report(summary, report)

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, interrupt)
    seed = max([row["seed"] for row in report["jobs"]] +
               [int(path.name[4:]) for path in output.glob("seed*")
                if path.is_dir() and path.name[4:].isdigit()] + [seed_start - 1]) + 1
    consecutive = 0
    try:
        save()
        while report["success_count"] < target:
            remaining = max_attempts - len(report["jobs"])
            if remaining <= 0:
                report.update(status="needs_attention", reason=f"{max_attempts} attempts exhausted")
                break
            if seed >= SEED_LIMIT:
                report.update(status="needs_attention", reason="Unsigned 32-bit seed range exhausted")
                break
            count = min(size, remaining, SEED_LIMIT - seed) if continuous else 1
            jobs = [{"suite": suite, "task_id": task_id, "seed": value}
                    for value in range(seed, seed + count)]
            completed = list(report["jobs"])
            def progress(rows):
                report["jobs"] = completed + rows
                save()
            options = (dict(success_target=min(count, target - report["success_count"]), progress=progress)
                       if continuous and count > 1 else {})
            result = collect_parallel(jobs, output / f"seed{seed:04d}", workers=1,
                                      timeout=max(900., default_timeout(jobs, backend)),
                                      camera_resolution=(448, 448),
                                      episodes_per_worker=size, backend=backend, vision_crf=vision_crf, **options)
            report["jobs"] = completed + result["jobs"]
            for row in result["jobs"]:
                # A worker stops after five misses to rebuild its simulation.
                # Known physical rejections still consume seeds, but should not
                # abort replenishment before the configured attempt budget.
                consecutive = (0 if row["status"] == "success" or _physical_rejection(row)
                               else consecutive + 1)
            seed = max(row["seed"] for row in result["jobs"]) + 1
            save()
            print(f"[PROGRESS] {suite}:{task_id} {report['success_count']}/{target}", flush=True)
            if report["success_count"] < target and consecutive >= 5:
                report.update(status="needs_attention", reason="Five consecutive infrastructure or data failures")
                break
        else:
            report["status"] = "complete"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        raise
    except BaseException as error:
        report.update(status="error", error=repr(error))
        raise
    finally:
        save()
        signal.signal(signal.SIGTERM, previous)


def archive_collection(source, destination):
    """Copy an inactive collection and keep its old directory as a backup."""
    source, destination = Path(source), Path(destination).resolve()
    if source.resolve() == destination.resolve() and not source.is_symlink():
        raise ValueError("Archive source and destination must differ")
    if source.is_symlink():
        if source.resolve() != destination.resolve():
            raise RuntimeError(f"Unexpected archive link: {source}")
        return {"source": str(source), "destination": str(destination), "verified": True}
    if source.resolve().is_relative_to(destination) or destination.is_relative_to(source.resolve()):
        raise ValueError("Archive source and destination must not overlap")
    destination.mkdir(parents=True, exist_ok=True)
    command = ["rsync", "-a", str(source) + "/", str(destination) + "/"]
    subprocess.run(command, check=True, capture_output=True, text=True)
    result = subprocess.run(["rsync", "-a", "--dry-run", "--itemize-changes",
                             str(source) + "/", str(destination) + "/"],
                            check=True, capture_output=True, text=True)
    if result.stdout.strip():
        raise RuntimeError(f"Archive copy is incomplete: {source}: {result.stdout}")
    backup = source.with_name(f".{source.name}.pre_ssd")
    if backup.exists():
        raise FileExistsError(backup)
    source.rename(backup)
    try:
        source.symlink_to(destination, target_is_directory=True)
    except BaseException:
        backup.rename(source)
        raise
    return {"source": str(source), "destination": str(destination),
            "backup": str(backup), "verified": True}


def run_queue(config_path):
    """Adopt current workers; refill their slots immediately after task completion."""
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    vision_crf = config.get("vision_crf", 0)
    backend = config.get("backend", "gsmini")
    jobs = [(job["suite"], job["task_id"]) for job in config["tasks"]]
    adopted = [(slot["suite"], slot["task_id"]) for slot in config["adopt"].values() if slot]
    if len(set(jobs + adopted)) != len(jobs + adopted):
        raise ValueError("Queued and adopted tasks must be distinct")
    target = config.get("target", 100)
    seed_start = config.get("seed_start", 0)
    max_attempts = config.get("max_attempts", DEFAULT_MAX_ATTEMPTS)
    for slot in config["adopt"].values():
        if slot:
            _resume_report(Path(slot["output"]) / "summary.json", slot["suite"], slot["task_id"],
                           backend, task_summary=True)
    devices = _queue_devices(config["adopt"])
    root = Path(config["output_root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with ExitStack() as resources:
        lock = resources.enter_context((root / ".queue.lock").open("a"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = root / "queue_state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {
            "backend": backend, "vision_crf": vision_crf, "slots": config["adopt"], "pending": config["tasks"],
            "completed": [], "needs_attention": []}
        if state.get("backend") != backend:
            raise ValueError(f"Cannot resume queue with a different or missing backend: {state_path}")
        devices.update(_queue_devices(state["slots"]))
        state.update(status="running", backend=backend, pid=os.getpid(), config=str(config_path))
        state.pop("error", None)
        processes, archives = {}, {}
        pool = resources.enter_context(ThreadPoolExecutor(max_workers=2))
        try:
            # Include completed historical collections without occupying a GPU slot.
            for entry in config.get("archive", []):
                source, destination = entry["source"], entry["destination"]
                archives[source] = pool.submit(archive_collection, source, destination)
            while True:
                for gpu, slot in list(state["slots"].items()):
                    if not slot:
                        continue
                    handle = processes.get(slot["pid"])
                    if handle is not None:
                        handle.poll()
                    if process_alive(slot["pid"]):
                        try:
                            command = Path(f"/proc/{slot['pid']}/cmdline").read_bytes().decode()
                        except FileNotFoundError:
                            command = ""
                        if slot["output"] in command:
                            continue
                    summary = Path(slot["output"]) / "summary.json"
                    report = json.loads(summary.read_text()) if summary.exists() else {"status": "error"}
                    entry = {**slot, "status": report["status"], "success_count": report.get("success_count", 0)}
                    if report["status"] == "complete" and report.get("success_count") == report.get("target", 100):
                        state["completed"].append(entry)
                    else:
                        state["needs_attention"].append(entry)
                    processes.pop(slot["pid"], None)
                    source = slot["output"]
                    if slot.get("archive_to") and source not in archives:
                        archives[source] = pool.submit(archive_collection, source, slot["archive_to"])
                    state["slots"][gpu] = None
                blocked = set()
                for path in config.get("wait_for_queues", ()):
                    foreign_slots = json.loads(Path(path).read_text())["slots"]
                    blocked.update(str(gpu) for gpu, slot in foreign_slots.items()
                                   if slot and process_alive(slot["pid"]))
                free = {}
                if (config.get("minimum_free_memory_mib") or blocked
                        and any(value.startswith("GPU-") for value in (*blocked, *devices.values()))):
                    inventory = subprocess.check_output([
                        "nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"], text=True)
                    for line in inventory.splitlines():
                        index, uuid, memory = (part.strip() for part in line.split(","))
                        free.update({index: int(memory), uuid: int(memory)})
                        if index in blocked or any(uuid.startswith(value) for value in blocked if value.startswith("GPU-")):
                            blocked.update((index, uuid))
                            blocked.update(gpu for gpu, selected in devices.items()
                                           if selected == index or uuid.startswith(selected))
                for gpu, slot in state["slots"].items():
                    if slot or not state["pending"]:
                        continue
                    if (gpu in blocked or devices[gpu] in blocked or (config.get("minimum_free_memory_mib")
                            and free.get(devices[gpu], 0) < config["minimum_free_memory_mib"])):
                        continue
                    if config.get("max_workers") and sum(bool(value) for value in state["slots"].values()) >= config["max_workers"]:
                        break
                    preferred = config.get("preferred_suites", {}).get(gpu)
                    index = next((i for i, job in enumerate(state["pending"]) if job["suite"] == preferred), 0)
                    job = state["pending"][index]
                    output = root / f"{job['suite']}{job['task_id']}"
                    command = [sys.executable, "-u", "-m", "benchmark.expert.collection_queue", "--task",
                               job["suite"], str(job["task_id"]), "--gpu", devices[gpu], "--output", str(output)]
                    for option, value, default in (("--target", target, 100), ("--seed-start", seed_start, 0),
                                                    ("--max-attempts", max_attempts, DEFAULT_MAX_ATTEMPTS)):
                        if value != default:
                            command += [option, str(value)]
                    if vision_crf:
                        command += ["--vision-crf", str(vision_crf)]
                    logpath = root / f"gpu{gpu}_{job['suite']}{job['task_id']}.log"
                    with logpath.open("ab") as log:
                        handle = subprocess.Popen(command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": devices[gpu]},
                                                  stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                                  start_new_session=True)
                    state["pending"].pop(index)
                    processes[handle.pid] = handle
                    state["slots"][gpu] = {**job, "gpu": devices[gpu], "pid": handle.pid,
                                           "output": str(output), "log": str(logpath), "vision_crf": vision_crf}
                    write_json(state_path, state)
                    print(f"[QUEUE START] gpu={gpu} {job['suite']}:{job['task_id']} pid={handle.pid}", flush=True)
                state["archives"] = {}
                for source, future in archives.items():
                    if not future.done():
                        state["archives"][source] = {"status": "copying"}
                    else:
                        try:
                            state["archives"][source] = {"status": "complete", **future.result()}
                        except Exception as error:
                            state["archives"][source] = {"status": "error", "error": repr(error)}
                state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                write_json(state_path, state)
                if (not state["pending"] and not any(state["slots"].values())
                        and all(row["status"] != "copying" for row in state["archives"].values())):
                    archive_failed = any(row["status"] == "error" for row in state["archives"].values())
                    state["status"] = "needs_attention" if state["needs_attention"] or archive_failed else "complete"
                    break
                time.sleep(5)
        except KeyboardInterrupt:
            state["status"] = "interrupted"
            raise
        except BaseException as error:
            state.update(status="error", error=repr(error))
            raise
        finally:
            write_json(state_path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--task", nargs=2, metavar=("SUITE", "TASK_ID"))
    parser.add_argument("--gpu", help="Physical GPU index or UUID/prefix within CUDA_VISIBLE_DEVICES")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0, help="First seed for a new collection (default: 0)")
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS,
                        help="Total attempt budget including existing attempts (default: 500)")
    parser.add_argument("--vision-crf", type=int,
                        help="Visual RGB compression for --task: 0=lossless, 12=compact. For --config set vision_crf in JSON.")
    args = parser.parse_args()
    if args.config:
        if args.vision_crf is not None:
            parser.error("Set vision_crf in the queue JSON when using --config")
        if args.target != 100 or args.seed_start != 0 or args.max_attempts != DEFAULT_MAX_ATTEMPTS:
            parser.error("Set target, seed_start and max_attempts in the queue JSON when using --config")
        run_queue(args.config)
    else:
        suite = args.task[0]
        try:
            task_id = int(args.task[1])
        except ValueError:
            parser.error("Task ID must be an integer")
        if args.gpu is None or args.output is None:
            parser.error("GPU and output are required for task collection")
        try:
            gpu = selected_gpu(args.gpu)
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            parser.error(str(error))
        options = {"target": args.target} if args.target != 100 else {}
        if args.seed_start:
            options["seed_start"] = args.seed_start
        if args.max_attempts != DEFAULT_MAX_ATTEMPTS:
            options["max_attempts"] = args.max_attempts
        if args.vision_crf is not None:
            options["vision_crf"] = args.vision_crf
        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        try:
            collect_task(suite, task_id, gpu, args.output, **options)
        finally:
            if previous is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous


if __name__ == "__main__":
    main()
