#!/usr/bin/env python3
"""Prepare a DM05 vision FP16 engine and prefix MLP FP8 engines offline.

Calibration applies only to the prefix MLP FP8 engines. The vision engine is
built in FP16.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import random
import shlex
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

_DEFAULT_PREFIX_BUCKETS = (576, 704, 768, 896, 1024)
_MAX_FP8_PREFIX_CAPACITY = 1024


def _parser(*, legacy_alias: bool = False) -> argparse.ArgumentParser:
    description = __doc__
    if legacy_alias:
        description = (
            "Legacy prepare_fast_fp8 alias. It prepares a vision FP16 engine and "
            "prefix MLP FP8 engines; calibration applies only to the prefix MLP. "
            "Omitting --prefix-mlp-precision defaults to fp8. New commands should "
            "use python -m opendm.infer.prepare_fast."
        )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--prefix-mlp-precision",
        choices=("fp8",),
        required=not legacy_alias,
        default="fp8" if legacy_alias else None,
        help=(
            "Prefix MLP engine precision (currently only fp8 is supported; "
            "legacy alias default: fp8)."
            if legacy_alias
            else "Prefix MLP engine precision (currently only fp8 is supported)."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--inputs", type=Path, nargs="+")
    source.add_argument("--dataset-root", type=Path)
    source.add_argument("--jsonl", type=Path, nargs="+")
    source.add_argument("--requests-jsonl", type=Path)
    parser.add_argument("--media-root", type=Path, action="append", default=[])
    parser.add_argument(
        "--playground", choices=("libero", "robotwin2"), default="libero"
    )
    parser.add_argument("--playground-module")
    parser.add_argument("--dataset-name")
    parser.add_argument(
        "--config-override",
        action="append",
        default=[],
        metavar="DOTTED_PATH=JSON",
        help="Override a playground dataclass value, for example model_config.chunk_size=10.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=128,
        help="Deterministic calibration sample count (default: 128; increase it to cover every task).",
    )
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--prefix-capacity",
        type=int,
        default=1024,
        help="FP8 prefix capacity (maximum supported: 1024).",
    )
    parser.add_argument(
        "--bucket",
        type=int,
        default=1024,
        help="Calibration prefix bucket; must not exceed --prefix-capacity.",
    )
    parser.add_argument("--min-seq-len", type=int)
    parser.add_argument("--opt-seq-len", type=int)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--vision-workspace-gb", type=float, default=8.0)
    parser.add_argument("--prefix-workspace-gb", type=float, default=8.0)
    parser.add_argument("--keep-onnx", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Build a new generation even if the complete request is cached.",
    )
    return parser


def _output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.expanduser().absolute()
    return args.checkpoint.expanduser().absolute() / ".opendm" / "fast_fp8"


def _profile_lengths(args: argparse.Namespace) -> tuple[int, int]:
    if not 0 < args.prefix_capacity <= _MAX_FP8_PREFIX_CAPACITY:
        raise ValueError(
            "--prefix-capacity must be in [1, 1024]; FP8 preparation does not "
            "support larger capacities such as 2048."
        )
    if not 0 < args.bucket <= args.prefix_capacity:
        raise ValueError("--bucket must be within --prefix-capacity.")
    if args.playground_module and (
        args.min_seq_len is None or args.opt_seq_len is None
    ):
        raise ValueError(
            "Custom playground modules require explicit --min-seq-len and --opt-seq-len."
        )
    minimum = args.min_seq_len
    optimum = args.opt_seq_len
    if minimum is None:
        minimum = 896 if args.playground == "robotwin2" else 576
    if optimum is None:
        optimum = 896 if args.playground == "robotwin2" else 704
    if not 0 < minimum <= optimum <= args.prefix_capacity:
        raise ValueError(
            "Expected 0 < min-seq-len <= opt-seq-len <= prefix-capacity, got "
            f"{minimum}, {optimum}, {args.prefix_capacity}."
        )
    return minimum, optimum


def _prepared_buckets(
    configured,
    *,
    min_seq_len: int,
    prefix_capacity: int,
    num_images: int,
    mm_tokens_per_image: int,
) -> tuple[list[int], int]:
    minimum_prefix_len = num_images * (mm_tokens_per_image + 1)
    profile_minimum = max(min_seq_len, minimum_prefix_len)
    if configured is None:
        buckets = [
            value
            for value in _DEFAULT_PREFIX_BUCKETS
            if profile_minimum <= value <= prefix_capacity
        ]
    else:
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in configured
        ):
            raise ValueError("Explicit prefix buckets must contain integers only.")
        buckets = list(configured)
        if buckets != sorted(set(buckets)):
            raise ValueError(
                "Explicit prefix buckets must be strictly increasing and unique."
            )
        invalid = [
            value
            for value in buckets
            if not profile_minimum <= value <= prefix_capacity
        ]
        if invalid:
            raise ValueError(
                "Explicit prefix buckets must fit the FP8 engine profile "
                f"and image-token minimum [{profile_minimum}, "
                f"{prefix_capacity}], got {invalid}."
            )
    if not buckets:
        raise ValueError(
            "No configured fast prefix bucket fits the FP8 engine profile and "
            f"image-token minimum {profile_minimum}."
        )
    return buckets, minimum_prefix_len


def _run(command: list[str]) -> None:
    print(f"+ {shlex.join(command)}", flush=True)
    subprocess.run(command, check=True)


def _stage_command(stage: str, public_argv: list[str], staging: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "opendm.infer.prepare_fast",
        f"_{stage}",
        "--staging-dir",
        str(staging),
        *public_argv,
    ]


def _normalized_public_argv(
    args: argparse.Namespace, public_argv: list[str]
) -> list[str]:
    """Carry one explicit precision selection into the preprocessing process."""
    remaining = []
    index = 0
    while index < len(public_argv):
        value = public_argv[index]
        if value == "--prefix-mlp-precision":
            index += 2
            continue
        if value.startswith("--prefix-mlp-precision="):
            index += 1
            continue
        remaining.append(value)
        index += 1
    return ["--prefix-mlp-precision", args.prefix_mlp_precision, *remaining]


def _orchestrate(args: argparse.Namespace, public_argv: list[str]) -> None:
    min_seq, opt_seq = _profile_lengths(args)
    if args.samples < 2:
        raise ValueError("--samples must be at least 2 for prefix MLP FP8 calibration.")
    output = _output_dir(args)
    output.mkdir(parents=True, exist_ok=True)
    staging = output / f".building-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        normalized_argv = _normalized_public_argv(args, public_argv)
        _run(_stage_command("preprocess", normalized_argv, staging))
        status = json.loads((staging / "stage_status.json").read_text())
        if status.get("reused"):
            print(
                f"Reusing prepared vision FP16 and prefix MLP FP8 artifacts: "
                f"{output / 'prepared.json'}"
            )
            return
        metadata = json.loads((staging / "prepare_metadata.json").read_text())
        deployment = metadata["deployment"]
        num_images = deployment["num_images"]
        vision_command = [
            sys.executable,
            "-m",
            "opendm.infer.build_vision_trt",
            "--checkpoint",
            str(args.checkpoint),
            "--onnx-path",
            str(staging / "vision.onnx"),
            "--engine-path",
            str(staging / "vision.engine"),
            "--num-images",
            str(num_images),
            "--workspace-gb",
            str(args.vision_workspace_gb),
        ]
        if args.keep_onnx:
            vision_command.append("--keep-onnx")
        _run(vision_command)
        _run(
            [
                sys.executable,
                "-m",
                "opendm.infer.collect_prefix_mlp_calibration",
                "--checkpoint",
                str(args.checkpoint),
                "--vision-trt-engine-path",
                str(staging / "vision.engine"),
                "--inputs",
                str(staging / "calibration_inputs.pt"),
                "--output",
                str(staging / "calibration.json"),
                "--num-images",
                str(num_images),
                "--bucket",
                str(args.bucket),
                "--chunk-size",
                str(deployment["chunk_size"]),
                "--diffusion-steps",
                str(args.diffusion_steps),
                "--seed",
                str(args.seed),
            ]
        )
        _run(
            [
                sys.executable,
                "-m",
                "opendm.infer.build_prefix_mlp_fp8_trt",
                "--checkpoint",
                str(args.checkpoint),
                "--calibration-json",
                str(staging / "calibration.json"),
                "--output-dir",
                str(staging / "prefix_mlp_fp8"),
                "--min-seq-len",
                str(min_seq),
                "--opt-seq-len",
                str(opt_seq),
                "--max-seq-len",
                str(args.prefix_capacity),
                "--workspace-gb",
                str(args.prefix_workspace_gb),
            ]
        )
        _publish(staging, output, metadata)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _publish(staging: Path, output: Path, metadata: dict) -> None:
    from opendm.infer.fast_fp8_prepared import (
        PREPARED_FORMAT_VERSION,
        artifact_record,
        canonical_json,
        write_json_atomic,
    )

    required = [
        staging / "vision.engine",
        staging / "calibration.json",
        staging / "calibration_inputs.pt",
        staging / "sampling_manifest.json",
        staging / "prefix_mlp_fp8" / "prefix_mlp_quant_manifest.json",
    ]
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"Build did not produce required artifact: {path}")
    identity = hashlib.sha256(
        canonical_json(metadata["invocation_identity"])
        + (staging / "vision.engine").read_bytes()
        + (staging / "prefix_mlp_fp8" / "prefix_mlp_quant_manifest.json").read_bytes()
    ).hexdigest()[:20]
    generations = output / "generations"
    generations.mkdir(exist_ok=True)
    generation = generations / f"{identity}-{uuid.uuid4().hex[:8]}"
    staging.rename(generation)
    ready = {
        "format_version": PREPARED_FORMAT_VERSION,
        "kind": "dm05_fast_fp8_prepared",
        "checkpoint_provenance": str(metadata["checkpoint_provenance"]),
        "deployment": metadata["deployment"],
        "preprocessing": metadata["preprocessing"],
        "invocation_identity": metadata["invocation_identity"],
        "artifacts": {
            "vision_engine": artifact_record(output, generation / "vision.engine"),
            "prefix_manifest": artifact_record(
                output, generation / "prefix_mlp_fp8" / "prefix_mlp_quant_manifest.json"
            ),
            "calibration": artifact_record(output, generation / "calibration.json"),
            "calibration_inputs": artifact_record(
                output, generation / "calibration_inputs.pt"
            ),
            "sampling_manifest": artifact_record(
                output, generation / "sampling_manifest.json"
            ),
        },
    }
    write_json_atomic(output / "prepared.json", ready)
    if metadata.get("custom_output"):
        runtime_option = "--inference-config.prepared-fast-artifacts-dir " + str(output)
    else:
        runtime_option = "(no extra artifact option; fast discovers this checkpoint-local generation)"
    print(f"Prepared vision FP16 engine and prefix MLP FP8 engines: {output}")
    print(f"Runtime: --inference-config.backend fast {runtime_option}")
    print(
        "Preparation verifies artifact compatibility only; it is not task success-rate certification."
    )


def _load_module(value: str):
    candidate = Path(value)
    if candidate.suffix == ".py" or candidate.exists():
        spec = importlib.util.spec_from_file_location(
            "opendm_prepare_playground", candidate
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import playground file: {candidate}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(value)


def _decode_override(value: str):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _apply_override(root, expression: str) -> None:
    if "=" not in expression:
        raise ValueError(f"Expected dotted.path=JSON override, got {expression!r}")
    dotted, raw = expression.split("=", 1)
    if dotted == "model_config.model_name_or_path":
        raise ValueError(
            "model_config.model_name_or_path cannot be overridden; use --checkpoint so every stage has one identity."
        )
    target = root
    parts = dotted.split(".")
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise ValueError(f"Unknown config override path: {dotted}")
        target = getattr(target, part)
    if not hasattr(target, parts[-1]):
        raise ValueError(f"Unknown config override path: {dotted}")
    setattr(target, parts[-1], _decode_override(raw))


def _configured_experiment(args: argparse.Namespace):
    module_name = args.playground_module or f"playground.dm05_{args.playground}"
    module = _load_module(module_name)
    exp = module.DM05Exp()
    exp.model_config.model_name_or_path = str(args.checkpoint.expanduser().absolute())
    if args.dataset_name is not None:
        exp.data_config.dataset_name = args.dataset_name
    for expression in args.config_override:
        _apply_override(exp, expression)
    exp.inference_config.diffusion_steps = args.diffusion_steps
    exp.inference_config.backend = "fast"
    exp.inference_config.fast_prefix_len = args.prefix_capacity
    if (args.checkpoint / "adapter_config.json").is_file() or exp.use_lora:
        raise ValueError(
            "FP8 preparation requires a merged/exported checkpoint, not a LoRA adapter."
        )
    if exp.data_config.is_history:
        raise ValueError("Prepared Prefix MLP FP8 does not support history inputs.")
    if exp.model_config.precision_policy != "bf16_mixed" or not exp.model_config.bf16:
        raise ValueError(
            "Prepared Prefix MLP FP8 requires the BF16 mixed-precision model configuration."
        )
    return exp


def _source_files(args: argparse.Namespace) -> list[Path]:
    if args.inputs:
        return [path.expanduser().absolute() for path in args.inputs]
    if args.requests_jsonl:
        return [args.requests_jsonl.expanduser().absolute()]
    if args.jsonl:
        files = []
        for item in args.jsonl:
            path = item.expanduser().absolute()
            files.extend(sorted(path.rglob("*.jsonl")) if path.is_dir() else [path])
        return files
    root = args.dataset_root.expanduser().absolute()
    return sorted(root.rglob("*.jsonl"))


def _source_identity(paths: list[Path]) -> list[dict]:
    from opendm.infer.fast_fp8_prepared import sha256_file

    records = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def _read_cached_ready(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _invocation_identity(
    args: argparse.Namespace,
    *,
    deployment: dict,
    preprocessing: dict,
    checkpoint: list[dict],
    sources: list[dict],
    selected_media: list[dict] | None,
    min_seq_len: int,
    opt_seq_len: int,
) -> dict:
    """Build the existing FP8 cache key without CLI-only naming fields."""
    return {
        "deployment": deployment,
        "preprocessing": preprocessing,
        "checkpoint": checkpoint,
        "sources": sources,
        "selected_media": selected_media,
        "samples": args.samples,
        "seed": args.seed,
        "bucket": args.bucket,
        "min_seq_len": min_seq_len,
        "opt_seq_len": opt_seq_len,
        "vision_workspace_gb": args.vision_workspace_gb,
        "prefix_workspace_gb": args.prefix_workspace_gb,
        "keep_onnx": args.keep_onnx,
        "config_overrides": args.config_override,
        "playground": args.playground_module or args.playground,
        "dataset_name": args.dataset_name,
    }


def _checkpoint_identity(checkpoint: Path) -> list[dict]:
    from opendm.infer.fast_fp8_prepared import sha256_file

    checkpoint = checkpoint.expanduser().absolute()
    names = {
        "adapter_config.json",
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    }
    suffixes = (".safetensors", ".bin")
    paths = [
        path
        for path in sorted(checkpoint.rglob("*"))
        if path.is_file() and (path.name in names or path.name.endswith(suffixes))
    ]
    if not paths:
        raise ValueError(f"No checkpoint weight/config files found under {checkpoint}")
    return [
        {
            "path": path.relative_to(checkpoint).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]


def _sample_jsonl(paths: list[Path], count: int, seed: int) -> list[dict]:
    # Keep a small deterministic reservoir per task/episode. This bounds row
    # memory while retaining task and episode coverage on large datasets.
    groups: dict[str, dict[str, list[dict]]] = {}
    seen: dict[tuple[str, str], int] = {}
    reservoir_size = count
    randomizer = random.Random(seed)
    for path in paths:
        with path.open() as stream:
            for line_index, line in enumerate(stream):
                if not line.strip():
                    continue
                record = json.loads(line)
                prompt = str(record.get("prompt", record.get("text", "")))
                item = {
                    "path": path,
                    "line": line_index,
                    "record": record,
                    "prompt": prompt,
                }
                episode = str(
                    record.get(
                        "episode_id",
                        record.get("episode", record.get("trajectory_id", path)),
                    )
                )
                key = (prompt, episode)
                seen[key] = seen.get(key, 0) + 1
                reservoir = groups.setdefault(prompt, {}).setdefault(episode, [])
                if len(reservoir) < reservoir_size:
                    reservoir.append(item)
                else:
                    replacement = randomizer.randrange(seen[key])
                    if replacement < reservoir_size:
                        reservoir[replacement] = item
    if not groups:
        raise ValueError("No observations were found in the selected JSONL files.")
    task_queues: dict[str, list[dict]] = {}
    for task in sorted(groups):
        episodes = groups[task]
        episode_queues = []
        for episode in sorted(episodes):
            values = episodes[episode]
            randomizer.shuffle(values)
            episode_queues.append(values)
        task_values = []
        while episode_queues:
            remaining = []
            for queue in episode_queues:
                task_values.append(queue.pop())
                if queue:
                    remaining.append(queue)
            episode_queues = remaining
        task_queues[task] = task_values
    chosen = []
    active_tasks = sorted(task_queues)
    randomizer.shuffle(active_tasks)
    while active_tasks and len(chosen) < count:
        remaining_tasks = []
        for task in active_tasks:
            if len(chosen) >= count:
                break
            chosen.append(task_queues[task].pop(0))
            if task_queues[task]:
                remaining_tasks.append(task)
        active_tasks = remaining_tasks
    if len(chosen) < 2:
        raise ValueError("At least two observations are required.")
    return chosen


def _chosen_media_identity(
    args, chosen: list[dict], image_keys: list[str]
) -> list[dict]:
    from opendm.infer.fast_fp8_prepared import sha256_file

    paths: dict[str, Path] = {}
    selection = []
    for item in chosen:
        record = item["record"]
        values = (
            record.get("images")
            if args.requests_jsonl
            else [record[key] for key in image_keys]
        )
        if not isinstance(values, list):
            raise ValueError(
                "Raw requests must contain an images path/descriptor list."
            )
        selected = []
        for value in values:
            descriptor = {"url": value} if isinstance(value, str) else value
            path = _resolve_media_path(str(descriptor["url"]), item["path"], args)
            paths[str(path)] = path
            selected.append(
                {
                    "path": str(path),
                    "type": descriptor.get("type", "image"),
                    "frame_idx": descriptor.get("frame_idx"),
                }
            )
        selection.append(
            {"jsonl": str(item["path"]), "line": item["line"] + 1, "media": selected}
        )
    files = [
        {"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for name, path in sorted(paths.items())
    ]
    return [{"selection": selection, "files": files}]


def _resolve_media_path(url: str, jsonl_path: Path, args: argparse.Namespace) -> Path:
    path = Path(url).expanduser()
    if path.is_absolute():
        candidates = [path]
    else:
        relative = Path(url[2:] if url.startswith("./") else url)
        roots = [jsonl_path.parent, *args.media_root]
        if args.dataset_root:
            dataset_root = args.dataset_root.expanduser().absolute()
            roots.extend([dataset_root, dataset_root / "image", dataset_root / "video"])
        candidates = [Path(root).expanduser().absolute() / relative for root in roots]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Cannot resolve media URL {url!r} from {jsonl_path}; tried {candidates}"
    )


def _decode_media(value, jsonl_path: Path, args: argparse.Namespace):
    from PIL import Image

    if isinstance(value, str):
        descriptor = {"type": "image", "url": value}
    elif isinstance(value, dict):
        descriptor = value
    else:
        raise ValueError(
            f"Image entry must be a path or descriptor, got {type(value)!r}"
        )
    path = _resolve_media_path(str(descriptor["url"]), jsonl_path, args)
    media_type = descriptor.get("type", "image")
    if media_type == "image":
        with Image.open(path) as image:
            return image.convert("RGB")
    if media_type == "video":
        if "frame_idx" not in descriptor:
            raise ValueError(f"Video descriptor is missing frame_idx: {descriptor}")
        import av

        target = descriptor["frame_idx"]
        if isinstance(target, bool) or not isinstance(target, int) or target < 0:
            raise ValueError("frame_idx must be a nonnegative integer")
        with av.open(str(path)) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index == target:
                    return frame.to_image().convert("RGB")
        raise IndexError(f"Video frame {target} does not exist: {path}")
    raise ValueError(f"Unsupported media type {media_type!r}")


def _records_to_tensors(args, exp, chosen: list[dict], image_keys: list[str]):
    tensors = []
    manifest = []
    for item in chosen:
        record = item["record"]
        if args.requests_jsonl:
            image_values = record.get("images")
            text = record.get("text", record.get("prompt", ""))
            states = record.get("states", record.get("state"))
        else:
            image_values = [record[key] for key in image_keys]
            text = record.get("prompt", record.get("text", ""))
            states = record.get("state", record.get("states"))
        if not isinstance(image_values, list):
            raise ValueError(
                "Raw requests must contain an images path/descriptor list."
            )
        images = [_decode_media(value, item["path"], args) for value in image_values]
        raw = exp.inference_config._prepare_model_input(
            text=text,
            images=images,
            states=states,
            robot_type=record.get("robot_type"),
            speed=record.get("speed"),
            control_mode=record.get("control_mode"),
        )
        transformed = exp.inference_config.input_transform(raw)
        sample = {
            key: transformed[key].detach().cpu()
            for key in ("pixel_values", "input_ids", "attention_mask", "token_type_ids")
        }
        if sample["input_ids"].shape[1] > args.bucket:
            raise ValueError(
                f"Sample {item['path']}:{item['line'] + 1} has {sample['input_ids'].shape[1]} "
                f"tokens, exceeding --bucket={args.bucket}; inputs are never truncated."
            )
        tensors.append(sample)
        manifest.append(
            {
                "jsonl": str(item["path"]),
                "line": item["line"] + 1,
                "task": item["prompt"],
                "prefix_tokens": int(sample["input_ids"].shape[1]),
                "media": [
                    str(value.get("url", "")) if isinstance(value, dict) else str(value)
                    for value in image_values
                ],
            }
        )
    return tensors, manifest


def _preprocess(args: argparse.Namespace, staging: Path) -> None:
    import torch

    from opendm.infer.fast_fp8_prepared import (
        load_prepared_fast_artifacts,
        preprocessing_contract,
    )

    exp = _configured_experiment(args)
    norm_stats_path, dataset_info = exp._inference_preprocessing_settings()
    preprocessing = preprocessing_contract(
        checkpoint=args.checkpoint,
        norm_stats_path=norm_stats_path,
        image_prompts=exp.inference_config.image_prompts,
        n_bins=exp.data_config.n_bins,
        add_state=exp.data_config.add_state,
        model_max_length=min(exp.trainer_config.model_max_length, args.prefix_capacity),
        clip_to_bounds=exp.inference_config.clip_to_bounds,
        compose_eef_rot=exp.inference_config.compose_eef_rot,
        default_robot_type=exp.inference_config._request_default("default_robot_type"),
        default_state_desc=exp.inference_config._request_default("default_state_desc"),
        default_speed=exp.inference_config._request_default("default_speed", "0.5"),
        default_control_mode=exp.inference_config._request_default(
            "default_control_mode"
        ),
    )
    num_images = len(exp.inference_config.image_prompts)
    if exp.data_config.is_history:
        num_images += int(exp.inference_config.max_history_images)
    from opendm.model.dm05.dm05_arch import DM05Config

    min_seq, opt_seq = _profile_lengths(args)
    model_config = DM05Config.from_pretrained(str(args.checkpoint))
    mm_tokens_per_image = int(model_config.vlm_config.mm_tokens_per_image)
    configured_buckets = exp.inference_config.prefix_seq_len_buckets
    buckets, _ = _prepared_buckets(
        configured_buckets,
        min_seq_len=min_seq,
        prefix_capacity=args.prefix_capacity,
        num_images=num_images,
        mm_tokens_per_image=mm_tokens_per_image,
    )
    deployment = {
        "num_images": num_images,
        "chunk_size": int(exp.model_config.chunk_size),
        "diffusion_steps": int(args.diffusion_steps),
        "prefix_capacity": int(args.prefix_capacity),
        "prefix_seq_len_buckets": buckets,
        "history": bool(exp.data_config.is_history),
    }
    paths = _source_files(args)
    source_identity = _source_identity(paths)
    image_keys = list(dataset_info.get("image_keys", []))
    chosen = None
    media_identity = None
    if not args.inputs:
        chosen = _sample_jsonl(paths, args.samples, args.seed)
        media_identity = _chosen_media_identity(args, chosen, image_keys)
    invocation = _invocation_identity(
        args,
        deployment=deployment,
        preprocessing=preprocessing,
        checkpoint=_checkpoint_identity(args.checkpoint),
        sources=source_identity,
        selected_media=media_identity,
        min_seq_len=min_seq,
        opt_seq_len=opt_seq,
    )
    output = _output_dir(args)
    ready_path = output / "prepared.json"
    if ready_path.is_file() and not args.force:
        ready = _read_cached_ready(ready_path)
        if ready is None:
            print("Cached prepared.json is invalid; rebuilding.", flush=True)
        if isinstance(ready, dict) and ready.get("invocation_identity") == invocation:
            from opendm.infer.fast_fp8_prepared import PreparedArtifactError

            try:
                prepared = load_prepared_fast_artifacts(
                    checkpoint=args.checkpoint,
                    prepared_dir=output,
                    preprocessing=preprocessing,
                    prefix_seq_len_buckets=buckets,
                    **{
                        key: deployment[key]
                        for key in (
                            "num_images",
                            "chunk_size",
                            "diffusion_steps",
                            "prefix_capacity",
                        )
                    },
                    is_history=deployment["history"],
                    verify_build_environment=True,
                )
            except PreparedArtifactError as exc:
                print(
                    f"Cached generation is incompatible; rebuilding: {exc}", flush=True
                )
            else:
                (staging / "stage_status.json").write_text(
                    json.dumps({"reused": True, "root": str(prepared.root)})
                )
                return
    if args.inputs:
        samples = []
        origins = []
        for path in paths:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            records = payload if isinstance(payload, list) else [payload]
            samples.extend(records)
            origins.extend(
                {"source_file": str(path), "original_index": index}
                for index in range(len(records))
            )
        if len(samples) < 2:
            raise ValueError("Tensor inputs contain fewer than two samples.")
        if args.samples < len(samples):
            indices = list(range(len(samples)))
            random.Random(args.seed).shuffle(indices)
            selected = indices[: args.samples]
            samples = [samples[index] for index in selected]
            origins = [origins[index] for index in selected]
        sampling_manifest = origins
    else:
        exp.initialize_inference_preprocessing(device="cpu")
        samples, sampling_manifest = _records_to_tensors(args, exp, chosen, image_keys)
    torch.save(samples, staging / "calibration_inputs.pt")
    (staging / "sampling_manifest.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "requested_samples": args.samples,
                "samples": sampling_manifest,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    metadata = {
        "checkpoint_provenance": str(args.checkpoint.expanduser().absolute()),
        "deployment": deployment,
        "preprocessing": preprocessing,
        "invocation_identity": invocation,
        "custom_output": args.output_dir is not None,
    }
    (staging / "prepare_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    (staging / "stage_status.json").write_text(json.dumps({"reused": False}))


def main(argv: list[str] | None = None, *, legacy_alias: bool = False) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    stage = None
    staging = None
    if argv and argv[0].startswith("_"):
        stage = argv.pop(0)
        internal = argparse.ArgumentParser(add_help=False)
        internal.add_argument("--staging-dir", type=Path, required=True)
        internal_args, argv = internal.parse_known_args(argv)
        staging = internal_args.staging_dir
    args = _parser(legacy_alias=legacy_alias).parse_args(argv)
    if stage is None:
        _orchestrate(args, argv)
    elif stage == "_preprocess":
        _preprocess(args, staging)
    else:
        raise ValueError(f"Unknown internal stage: {stage}")


if __name__ == "__main__":
    main()
