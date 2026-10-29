#!/usr/bin/env python3
"""Collect FP8 ranges from real DM05 transform outputs, without implicit builds.

Each --inputs file is a torch.save dictionary (or a list of dictionaries) with
pixel_values, input_ids, attention_mask and token_type_ids. Save these tensors
from the deployment transform after constructing prompt/state/image tokens;
this tool does not synthesize observations or truncate oversized requests.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from opendm.infer.dm05_infer import DM05FastInferRuntime
from opendm.infer.prefix_mlp_fp8_contract import (
    PREFIX_MLP_FORMAT_VERSION,
    SPEC,
    canonical_json,
    prefix_mlp_model_contract,
    prefix_source_model_sha256,
    sha256_file,
    utc8_now,
    validate_calibration,
    write_json_atomic,
)
from opendm.model.dm05.dm05_lora import load_dm05_model_for_inference, unwrap_dm05_model
from opendm.model.dm05.dm05_utils import mask_history_pad_tokens_in_attention

_REQUIRED_INPUTS = {"pixel_values", "input_ids", "attention_mask", "token_type_ids"}
_OPTIONAL_INPUTS = {"diffusion_input_noise", "action_mask"}


def _load_samples(paths: list[Path], *, bucket: int, num_images: int):
    samples, provenance, unique = [], [], set()
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        records = payload if isinstance(payload, list) else [payload]
        if not records:
            raise ValueError(f"Calibration input file contains no samples: {path}.")
        for index, record in enumerate(records):
            if not isinstance(record, dict) or not _REQUIRED_INPUTS <= record.keys():
                raise ValueError(
                    f"{path} sample {index} must contain {_REQUIRED_INPUTS}."
                )
            if record.keys() - (_REQUIRED_INPUTS | _OPTIONAL_INPUTS):
                raise ValueError(
                    f"Unsupported calibration keys in {path} sample {index}."
                )
            if not all(isinstance(value, torch.Tensor) for value in record.values()):
                raise ValueError("Calibration inputs must contain tensors only.")
            ids = record["input_ids"]
            if ids.ndim != 2 or ids.shape[0] != 1 or not 0 < ids.shape[1] <= bucket:
                raise ValueError(
                    f"Expected input_ids [1, 1..{bucket}], got {tuple(ids.shape)}; inputs are never truncated."
                )
            if ids.dtype not in (torch.int32, torch.int64) or bool((ids < 0).any()):
                raise ValueError(
                    "input_ids must contain nonnegative integer token IDs."
                )
            for key in ("attention_mask", "token_type_ids"):
                value = record[key]
                if value.shape != ids.shape or not bool(
                    ((value == 0) | (value == 1)).all()
                ):
                    raise ValueError(
                        f"{key} must be a binary tensor with the same shape as input_ids."
                    )
            if not bool(record["attention_mask"].any()):
                raise ValueError("Calibration sample has no visible tokens.")
            pixels = record["pixel_values"]
            if pixels.ndim != 4 or pixels.shape[:2] != (num_images, 3):
                raise ValueError(
                    f"Expected pixel_values [{num_images}, 3, H, W], got {tuple(pixels.shape)}."
                )
            if not pixels.is_floating_point() or not bool(torch.isfinite(pixels).all()):
                raise ValueError(
                    "pixel_values must contain finite transformed floating-point pixels."
                )
            digest = hashlib.sha256()
            for key in sorted(_REQUIRED_INPUTS):
                value = record[key].contiguous()
                digest.update(
                    canonical_json([key, list(value.shape), str(value.dtype)])
                )
                digest.update(memoryview(value.view(torch.uint8).numpy()).cast("B"))
            unique.add(digest.hexdigest())
            samples.append(record)
        provenance.append(
            {"path": str(path), "sha256": sha256_file(path), "samples": len(records)}
        )
    if len(unique) < 2:
        raise ValueError(
            "Calibration requires at least two distinct real transformed input samples."
        )
    return samples, provenance, len(unique)


def _accumulate(stats: dict, value: torch.Tensor, visible_mask: torch.Tensor) -> None:
    if value.ndim != 3 or value.shape[:2] != visible_mask.shape:
        raise ValueError(
            "Observed MLP activation does not match the prefix visible-row mask."
        )
    visible = value.detach()[visible_mask].float().abs()
    if visible.numel() == 0 or not bool(torch.isfinite(visible).all().item()):
        raise ValueError(
            "Calibration activations must be nonempty and finite on visible rows."
        )
    stats["max_abs"] = max(stats["max_abs"], float(visible.max().item()))
    stats["rows"] += int(visible.shape[0])
    stats["elements"] += int(visible.numel())


@torch.inference_mode()
def collect_calibration(args: argparse.Namespace) -> dict:
    if args.num_images <= 0 or not 0 < args.bucket <= 1024:
        raise ValueError("Expected positive --num-images and --bucket in 1..1024.")
    if args.chunk_size is not None and args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")
    if args.diffusion_steps <= 0:
        raise ValueError("--diffusion-steps must be positive.")
    samples, input_files, unique_count = _load_samples(
        args.inputs, bucket=args.bucket, num_images=args.num_images
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Collecting Prefix MLP calibration requires CUDA.")
    if not args.vision_trt_engine_path.is_file():
        raise FileNotFoundError(
            "Build the matching vision engine explicitly before calibration."
        )
    vision_record = {
        "file": args.vision_trt_engine_path.name,
        "bytes": args.vision_trt_engine_path.stat().st_size,
        "sha256": sha256_file(args.vision_trt_engine_path),
    }
    overrides = None if args.chunk_size is None else {"chunk_size": args.chunk_size}
    model = load_dm05_model_for_inference(
        str(args.checkpoint),
        config_overrides=overrides,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    dm05_model = unwrap_dm05_model(model)
    vlm = dm05_model.model.vlm
    language_model = vlm.model.language_model
    # Fingerprint before fast-runtime setup mutates execution-only model state.
    model_contract = prefix_mlp_model_contract(language_model)
    source_sha = prefix_source_model_sha256(vlm)
    dm05_model.set_attention_implementation(
        llm_attn_implementation="flex_attention",
        vision_attn_implementation="flash_attention_2",
        action_attn_implementation="sdpa",
        bf16=True,
    )
    dm05_model.enable_gradient_checkpointing(
        vlm_gradient_checkpointing=False,
        ae_gradient_checkpointing=False,
    )
    model = model.eval().to("cuda")
    runtime = DM05FastInferRuntime(
        dm05_model,
        vision_trt_engine_path=args.vision_trt_engine_path,
        num_current_images=args.num_images,
        prefix_seq_len_buckets=[args.bucket],
        diffusion_steps=args.diffusion_steps,
    )
    profile = runtime.graph_profiles[args.bucket]
    loaded_vision_identity = runtime.vision_trt_runner.engine_identity
    if any(
        loaded_vision_identity.get(key) != vision_record[key]
        for key in ("bytes", "sha256")
    ):
        raise ValueError(
            "Vision engine changed while calibration was starting; retry with a stable artifact."
        )
    vision_record = {"file": args.vision_trt_engine_path.name, **loaded_vision_identity}
    layers = {
        str(index): {
            activation: {"visible_rows": {"max_abs": 0.0, "rows": 0, "elements": 0}}
            for activation in ("mlp_input", "down_input")
        }
        for index in range(model_contract["layer_count"])
    }
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    noise_shape = (1, runtime.suffix_len, runtime.action_dim)
    for sample_index, sample in enumerate(samples):
        noise = sample.get("diffusion_input_noise")
        if noise is None:
            noise = torch.randn(noise_shape, generator=generator, dtype=torch.float32)
        action_mask = sample.get("action_mask")
        if action_mask is None:
            action_mask = torch.ones((1, 1, runtime.action_dim))
        if tuple(noise.shape) != noise_shape or not bool(torch.isfinite(noise).all()):
            raise ValueError(
                f"diffusion_input_noise must be finite with exact shape {noise_shape}; check --chunk-size."
            )
        if tuple(action_mask.shape) != (1, 1, runtime.action_dim) or not bool(
            torch.isfinite(action_mask).all()
        ):
            raise ValueError(
                "action_mask must be finite with exact shape [1, 1, action_dim]."
            )
        runtime.infer(
            images=sample["pixel_values"],
            input_ids=sample["input_ids"],
            attention_mask=sample["attention_mask"],
            token_type_ids=sample["token_type_ids"],
            diffusion_input_noise=noise,
            action_mask=action_mask,
            diffusion_steps=args.diffusion_steps,
        )
        visible_mask = mask_history_pad_tokens_in_attention(
            input_ids=profile.input_ids,
            attention_mask=profile.attention_mask,
        ).bool()
        handles, observed = [], set()

        def capture(
            index: int, activation: str, observed=observed, visible_mask=visible_mask
        ):
            def hook(_module, positional):
                key = (index, activation)
                if key in observed:
                    raise RuntimeError(
                        "Prefix MLP calibration hook ran more than once for one sample."
                    )
                observed.add(key)
                _accumulate(
                    layers[str(index)][activation]["visible_rows"],
                    positional[0],
                    visible_mask,
                )

            return hook

        for index, layer in enumerate(language_model.layers):
            handles.append(
                layer.mlp.register_forward_pre_hook(capture(index, "mlp_input"))
            )
            handles.append(
                layer.mlp.down_proj.register_forward_pre_hook(
                    capture(index, "down_input")
                )
            )
        try:
            # Hooks execute on the eager path after copying this sample into the
            # graph profile; graph replay itself does not invoke Python hooks.
            runtime._run_graph_profile_ops(profile)
            torch.cuda.synchronize(runtime.device)
        finally:
            for handle in handles:
                handle.remove()
        if len(observed) != model_contract["layer_count"] * 2:
            raise RuntimeError(
                "Failed to observe every Prefix MLP activation; no calibration was written."
            )
        print(
            f"calibration sample {sample_index + 1}/{len(samples)}: {int(visible_mask.sum().item())} visible tokens",
            flush=True,
        )
    result = {
        "format_version": PREFIX_MLP_FORMAT_VERSION,
        "spec": SPEC,
        "created_at": utc8_now(),
        "checkpoint": str(args.checkpoint),
        "model_contract": model_contract,
        "prefix_source_sha256": source_sha,
        "vision_engine": vision_record,
        "batch_size": 1,
        "num_images": args.num_images,
        "bucket": args.bucket,
        "sample_count": len(samples),
        "distinct_sample_count": unique_count,
        "input_files": input_files,
        "actual_prefix_lengths": [
            int(sample["input_ids"].shape[1]) for sample in samples
        ],
        "effective_config": {
            "chunk_size": runtime.suffix_len,
            "action_dim": runtime.action_dim,
            "diffusion_steps": args.diffusion_steps,
            "seed": args.seed,
            "history": False,
        },
        "layers": layers,
    }
    validate_calibration(result, model_contract, prefix_source_sha256=source_sha)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vision-trt-engine-path", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-images", type=int, required=True)
    parser.add_argument("--bucket", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260915)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = collect_calibration(args)
    write_json_atomic(args.output, result)
    print(f"Saved Prefix MLP calibration: {args.output}", flush=True)


if __name__ == "__main__":
    main()
