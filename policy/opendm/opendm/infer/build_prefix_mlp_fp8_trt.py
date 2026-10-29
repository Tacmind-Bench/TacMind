#!/usr/bin/env python3
"""Explicitly build a calibrated, checkpoint-bound DM05 FP8 Prefix MLP bundle.

This changes model numerics. Validate task success before deploying the bundle;
runtime loading never invokes this builder implicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import uuid
from pathlib import Path

import numpy as np
import torch

from opendm.infer.dm05_trt_utils import load_tensorrt
from opendm.infer.prefix_mlp_fp8_contract import (
    MANIFEST_NAME,
    PREFIX_MLP_FORMAT_VERSION,
    SPEC,
    bundle_lock,
    cleanup_bundle_tree,
    current_build_metadata,
    prefix_mlp_model_contract,
    prefix_source_model_sha256,
    publish_bundle_generation,
    read_activation_amax,
    sha256_file,
    utc8_now,
    validate_bundle_manifest,
    validate_calibration,
    write_json_atomic,
)
from opendm.model.dm05.dm05_lora import load_dm05_model_for_inference, unwrap_dm05_model

FP8_MAX = 448.0


def _scale_from_amax(amax: float) -> np.ndarray:
    if not math.isfinite(amax) or not 0 <= amax <= np.finfo(np.float16).max:
        raise ValueError(
            "FP8 QDQ requires finite ranges representable by its FP16 tensors."
        )
    return np.asarray([max(float(amax) / FP8_MAX, 1e-8)], dtype=np.float32)


def _add_fp8_qdq(
    *,
    trt,
    network,
    tensor,
    scale: np.ndarray,
    name: str,
    keepalive: list[np.ndarray],
):
    keepalive.append(scale)
    scale_layer = network.add_constant(tuple(scale.shape), scale)
    if scale_layer is None:
        raise RuntimeError(f"Failed to add FP8 scale for {name}.")
    scale_layer.name = f"{name}.scale"
    quantize = network.add_quantize(
        tensor,
        scale_layer.get_output(0),
        trt.fp8,
    )
    if quantize is None:
        raise RuntimeError(f"Failed to add FP8 quantize layer for {name}.")
    quantize.name = f"{name}.quantize"
    dequantize = network.add_dequantize(
        quantize.get_output(0),
        scale_layer.get_output(0),
        trt.float16,
    )
    if dequantize is None:
        raise RuntimeError(f"Failed to add FP8 dequantize layer for {name}.")
    dequantize.name = f"{name}.dequantize"
    return dequantize.get_output(0)


def _add_fp8_weight(
    *,
    trt,
    network,
    weight: torch.Tensor,
    name: str,
    keepalive: list[np.ndarray],
):
    weight_kn = (
        weight.detach()
        .to(dtype=torch.float16, device="cpu")
        .transpose(0, 1)
        .contiguous()
        .numpy()
    )
    batched_weight = weight_kn[None, ...]
    keepalive.append(batched_weight)
    constant = network.add_constant(tuple(batched_weight.shape), batched_weight)
    if constant is None:
        raise RuntimeError(f"Failed to add TensorRT weight for {name}.")
    constant.name = f"{name}.weight"
    return _add_fp8_qdq(
        trt=trt,
        network=network,
        tensor=constant.get_output(0),
        scale=_scale_from_amax(float(np.abs(weight_kn).max())),
        name=f"{name}.weight",
        keepalive=keepalive,
    )


def build_layer_engine(
    *,
    engine_path: Path,
    mlp,
    hidden_size: int,
    input_amax: float,
    down_amax: float,
    min_seq_len: int,
    opt_seq_len: int,
    max_seq_len: int,
    workspace_gb: float,
) -> None:
    trt = load_tensorrt("Building FP8 Prefix MLP TensorRT engines")
    if not hasattr(trt, "fp8"):
        raise RuntimeError("The installed TensorRT version does not support FP8.")
    if not hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"):
        raise RuntimeError("Building FP8 Prefix MLP requires strongly typed TensorRT.")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    hidden_states = network.add_input(
        "hidden_states",
        trt.float16,
        (1, -1, hidden_size),
    )
    if hidden_states is None:
        raise RuntimeError("Failed to add Prefix MLP hidden_states input.")

    keepalive: list[np.ndarray] = []
    mlp_input = _add_fp8_qdq(
        trt=trt,
        network=network,
        tensor=hidden_states,
        scale=_scale_from_amax(input_amax),
        name="mlp_input",
        keepalive=keepalive,
    )
    gate_weight = _add_fp8_weight(
        trt=trt,
        network=network,
        weight=mlp.gate_proj.weight,
        name="gate_proj",
        keepalive=keepalive,
    )
    up_weight = _add_fp8_weight(
        trt=trt,
        network=network,
        weight=mlp.up_proj.weight,
        name="up_proj",
        keepalive=keepalive,
    )
    down_weight = _add_fp8_weight(
        trt=trt,
        network=network,
        weight=mlp.down_proj.weight,
        name="down_proj",
        keepalive=keepalive,
    )

    gate = network.add_matrix_multiply(
        mlp_input,
        trt.MatrixOperation.NONE,
        gate_weight,
        trt.MatrixOperation.NONE,
    )
    up = network.add_matrix_multiply(
        mlp_input,
        trt.MatrixOperation.NONE,
        up_weight,
        trt.MatrixOperation.NONE,
    )
    if gate is None or up is None:
        raise RuntimeError("Failed to add Prefix MLP gate/up GEMMs.")
    gate.name = "gate_proj"
    up.name = "up_proj"
    activation = network.add_activation(
        gate.get_output(0),
        trt.ActivationType.GELU_TANH,
    )
    if activation is None:
        raise RuntimeError("Failed to add Prefix MLP GELU activation.")
    activation.name = "gelu_tanh"
    product = network.add_elementwise(
        activation.get_output(0),
        up.get_output(0),
        trt.ElementWiseOperation.PROD,
    )
    if product is None:
        raise RuntimeError("Failed to add Prefix MLP gate/up product.")
    product.name = "gate_up_product"
    down_input = _add_fp8_qdq(
        trt=trt,
        network=network,
        tensor=product.get_output(0),
        scale=_scale_from_amax(down_amax),
        name="down_input",
        keepalive=keepalive,
    )
    down = network.add_matrix_multiply(
        down_input,
        trt.MatrixOperation.NONE,
        down_weight,
        trt.MatrixOperation.NONE,
    )
    if down is None:
        raise RuntimeError("Failed to add Prefix MLP down GEMM.")
    down.name = "down_proj"
    output = down.get_output(0)
    output.name = "layer_out"
    network.mark_output(output)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        int(workspace_gb * (1 << 30)),
    )
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    profile = builder.create_optimization_profile()
    min_shape = (1, min_seq_len, hidden_size)
    opt_shape = (1, opt_seq_len, hidden_size)
    max_shape = (1, max_seq_len, hidden_size)
    profile.set_shape("hidden_states", min_shape, opt_shape, max_shape)
    actual_shapes = tuple(
        tuple(int(dim) for dim in shape) for shape in profile.get_shape("hidden_states")
    )
    if actual_shapes != (min_shape, opt_shape, max_shape):
        raise RuntimeError(
            "TensorRT rejected Prefix MLP profile "
            f"min={min_shape}, opt={opt_shape}, max={max_shape}."
        )
    if config.add_optimization_profile(profile) < 0:
        raise RuntimeError("Failed to add the Prefix MLP optimization profile.")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Failed to build FP8 Prefix MLP TensorRT engine.")
    engine_path.write_bytes(bytes(serialized))


@torch.inference_mode()
def build_bundle(args: argparse.Namespace) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("Building FP8 Prefix MLP TensorRT engines requires CUDA.")
    lengths = (args.min_seq_len, args.opt_seq_len, args.max_seq_len)
    if lengths != tuple(sorted(lengths)) or args.min_seq_len <= 0:
        raise ValueError(f"Expected positive min <= opt <= max, got {lengths}.")
    if not math.isfinite(args.workspace_gb) or args.workspace_gb <= 0:
        raise ValueError("--workspace-gb must be finite and positive.")

    calibration_bytes = args.calibration_json.read_bytes()
    calibration = json.loads(calibration_bytes)
    if not isinstance(calibration, dict):
        raise ValueError("Calibration JSON must be an object.")
    # The action chunk does not change the prefix MLP, but retain the effective
    # model configuration used when collecting the representative requests.
    effective_config = calibration.get("effective_config", {})
    overrides = (
        {"chunk_size": effective_config["chunk_size"]}
        if "chunk_size" in effective_config
        else None
    )
    model = load_dm05_model_for_inference(
        str(args.checkpoint),
        config_overrides=overrides,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    ).eval()
    vlm = unwrap_dm05_model(model).model.vlm
    language_model = vlm.model.language_model
    model_contract = prefix_mlp_model_contract(language_model)
    source_sha = prefix_source_model_sha256(vlm)
    validate_calibration(calibration, model_contract, prefix_source_sha256=source_sha)
    metadata = current_build_metadata()
    engine_dir = args.output_dir.absolute()
    if engine_dir.is_symlink():
        raise ValueError("--output-dir cannot be a symlink.")

    # Hold one lock across an entire generation. Readers hold the corresponding
    # shared lock until every engine has been verified and deserialized.
    with bundle_lock(engine_dir, exclusive=True):
        if engine_dir.exists():
            if not engine_dir.is_dir():
                raise ValueError("--output-dir must be a directory.")
            if not args.force:
                raise FileExistsError(
                    f"Bundle directory exists: {engine_dir}; use --force to rebuild."
                )
            if any(engine_dir.iterdir()) and not (engine_dir / MANIFEST_NAME).is_file():
                raise ValueError("Refusing to replace a non-bundle directory.")
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{engine_dir.name}.building-", dir=engine_dir.parent
            )
        )
        backup = engine_dir.parent / f".{engine_dir.name}.previous-{uuid.uuid4().hex}"
        try:
            (staging / "calibration.json").write_bytes(calibration_bytes)
            records = {}
            for index, layer in enumerate(language_model.layers):
                input_amax = read_activation_amax(calibration, index, "mlp_input")
                down_amax = read_activation_amax(calibration, index, "down_input")
                engine_path = staging / f"layer_{index:02d}.engine"
                build_layer_engine(
                    engine_path=engine_path,
                    mlp=layer.mlp,
                    hidden_size=model_contract["hidden_size"],
                    input_amax=input_amax,
                    down_amax=down_amax,
                    min_seq_len=args.min_seq_len,
                    opt_seq_len=args.opt_seq_len,
                    max_seq_len=args.max_seq_len,
                    workspace_gb=args.workspace_gb,
                )
                records[str(index)] = {
                    "engine": engine_path.name,
                    "bytes": engine_path.stat().st_size,
                    "sha256": sha256_file(engine_path),
                    "input_range": input_amax,
                    "down_range": down_amax,
                }
                print(
                    f"layer {index}: built {engine_path.name} ({engine_path.stat().st_size / 2**20:.1f} MiB)",
                    flush=True,
                )
            manifest = {
                "format_version": PREFIX_MLP_FORMAT_VERSION,
                "spec": SPEC,
                "created_at": utc8_now(),
                "checkpoint": str(args.checkpoint),
                "model_contract": model_contract,
                "prefix_source_sha256": source_sha,
                "contract": {
                    "batch_size": 1,
                    "hidden_size": model_contract["hidden_size"],
                    "min_seq_len": args.min_seq_len,
                    "opt_seq_len": args.opt_seq_len,
                    "max_seq_len": args.max_seq_len,
                    "layer_indices": list(range(model_contract["layer_count"])),
                },
                "calibration": {
                    "file": "calibration.json",
                    "bytes": len(calibration_bytes),
                    "sha256": hashlib.sha256(calibration_bytes).hexdigest(),
                },
                "build_metadata": metadata,
                "engines": records,
            }
            validate_bundle_manifest(
                manifest, model_contract, prefix_source_sha256=source_sha
            )
            write_json_atomic(staging / MANIFEST_NAME, manifest)
            publish_bundle_generation(staging, engine_dir, backup)
        finally:
            cleanup_bundle_tree(staging, purpose="incomplete bundle staging directory")
    return engine_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-seq-len", type=int, default=576)
    parser.add_argument("--opt-seq-len", type=int, default=768)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--workspace-gb", type=float, default=8.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(f"Built FP8 Prefix MLP bundle: {build_bundle(parse_args())}", flush=True)


if __name__ == "__main__":
    main()
