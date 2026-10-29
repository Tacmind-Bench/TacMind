"""Strongly typed TensorRT action expert; training stays in PyTorch."""

import logging
from time import perf_counter

import torch
from torch import nn

logger = logging.getLogger(__name__)


def require_tensorrt():
    try:
        import torch_tensorrt
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "action_backend=tensorrt requires Torch-TensorRT matching PyTorch and CUDA; "
            "see docs/usage.md. Use action_backend=torch for the ordinary backend."
        ) from exc
    return torch_tensorrt


def check_actions(actual, expected, *, bf16):
    """Bound compiler roundoff by one internal-dtype epsilon in max and L2 norm."""
    if actual.shape != expected.shape:
        raise RuntimeError("TensorRT action validation changed output shape")
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise RuntimeError("TensorRT action validation produced nonfinite values")
    expected = expected.float()
    delta = (actual.float() - expected).abs()
    delta_norm = delta.norm()
    expected_norm = expected.norm()
    error = float(delta.max())
    relative = float(delta_norm / expected_norm.clamp_min(1e-8))
    tolerance = torch.finfo(torch.bfloat16).eps if bf16 else 1e-5
    if error > tolerance * max(1.0, float(expected.abs().max())) or float(
        delta_norm
    ) > tolerance * max(1.0, float(expected_norm)):
        raise RuntimeError(
            f"TensorRT action validation failed: max_abs={error:g}, relative_l2={relative:g}; "
            "use action_backend=torch."
        )
    return dict(max_abs=error, relative_l2=relative, tolerance=tolerance)


class _Suffix(nn.Module):
    def __init__(self, expert):
        super().__init__()
        self.expert = expert

    def forward(self, x, mask, positions, condition, modulations, *kv):
        depth = len(kv) // 2
        return self.expert(
            suffix_embeds=x,
            attention_mask=mask,
            position_ids=positions,
            adarms_cond=condition,
            tactile_modulations=modulations,
            prefix_cache_keys=kv[:depth],
            prefix_cache_values=kv[depth:],
        )


class TensorRTSuffix:
    def __init__(self, policy, noise, flow, *, modulation_dtype):
        trt = require_tensorrt()
        started = perf_counter()
        self.policy = policy
        expert = policy.model.action_expert
        x = policy._action_input_proj(noise)
        mask, positions = policy._build_suffix_metadata(
            input_ids=flow["input_ids"],
            prefix_len=flow["prefix_len"],
            suffix_len=x.shape[1],
            device=x.device,
            dtype=x.dtype,
        )
        condition = policy._build_adarms_cond(
            torch.ones(
                noise.shape[0],
                device=noise.device,
                dtype=policy.model.action_in_proj.weight.dtype,
            )
        )
        self.zeros = torch.zeros(
            len(expert.layers),
            *noise.shape[:2],
            4 * x.shape[-1],
            device=noise.device,
            dtype=modulation_dtype,
        )
        keys, values = policy._extract_prefix_cache_tensors(flow["kv_cache"])
        args = (x, mask == 0, positions, condition, self.zeros, *keys, *values)
        exported = torch.export.export(_Suffix(expert).eval(), args, strict=False)
        # SDPA decomposition can change output strides. A traced no-op
        # contiguous() may be absent; reshape keeps the same logical layout.
        for node in exported.graph.nodes:
            if node.target == torch.ops.aten.view.default:
                node.target = torch.ops.aten.reshape.default
        exported.graph_module.recompile()
        self.module = trt.dynamo.compile(
            exported,
            inputs=args,
            use_explicit_typing=True,
            disable_tf32=True,
            require_full_compilation=True,
            optimization_level=3,
            workspace_size=1 << 30,
            max_aux_streams=0,
            use_python_runtime=False,
            device=trt.Device(gpu_id=noise.device.index),
        )
        self.build_seconds = perf_counter() - started
        logger.info("Built TensorRT action expert in %.2f s", self.build_seconds)

    def __call__(
        self,
        *,
        suffix_embeds,
        attention_mask,
        position_ids,
        past_key_values,
        adarms_cond,
        tactile_modulations=None,
        **kwargs,
    ):
        keys, values = self.policy._extract_prefix_cache_tensors(past_key_values)
        mask = (
            attention_mask
            if attention_mask.dtype == torch.bool
            else attention_mask == 0
        )
        with torch.cuda.device(suffix_embeds.device):
            return self.module(
                suffix_embeds,
                mask,
                position_ids,
                adarms_cond,
                self.zeros if tactile_modulations is None else tactile_modulations,
                *keys,
                *values,
            )
