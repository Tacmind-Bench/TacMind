"""Strict offline loading of a TacMind-0 policy checkpoint."""

from pathlib import Path

import torch

from tacmind0.model.modeling import (
    TacMind0Config,
    TacMind0ForConditionalGeneration,
)

from .contracts import validate_contract


def strict_policy_load(
    path: str | Path,
    *,
    dtype: torch.dtype = torch.bfloat16,
    overrides: dict | None = None,
) -> TacMind0ForConditionalGeneration:
    config = TacMind0Config.from_pretrained(path, local_files_only=True)
    if config.tactile_config is not None:
        validate_contract(config.tactile_config)
    for name, value in (overrides or {}).items():
        setattr(config, name, value)
    model, info = TacMind0ForConditionalGeneration.from_pretrained(
        path,
        config=config,
        dtype=dtype,
        local_files_only=True,
        output_loading_info=True,
    )
    failures = {
        key: info.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if info.get(key)
    }
    if failures:
        raise RuntimeError(f"Strict policy loading failed: {failures}")
    return model
