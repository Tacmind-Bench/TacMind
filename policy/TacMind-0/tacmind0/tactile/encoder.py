"""Offline reconstruction of the Tac-LeWM encoder; no world-model runtime required."""

import math
from pathlib import Path

import torch
from torch import nn

from .backbones import DinoV2RegisterBackbone
from .tactile_sequence import sequence_position_buffers, wrap_vit_tactile_sequence


def build_encoder(backbone_config: dict) -> nn.Module:
    if (
        backbone_config.get("patch_size") != 14
        or backbone_config.get("hidden_size") != 384
    ):
        raise ValueError("Tac-LeWM requires patch_size=14 and hidden_size=384")
    return wrap_vit_tactile_sequence(
        DinoV2RegisterBackbone(backbone_config),
        frame_size=42,
        patch_size=14,
        history_size=8,
        time_stride=5,
        canonicalize_right=True,
        causal_attention=True,
    )


class FrozenTactileEncoder(nn.Module):
    def __init__(self, backbone_config: dict) -> None:
        super().__init__()
        self.encoder = build_encoder(backbone_config)
        self.requires_grad_(False)
        self.train(False)

    def rebuild_position_buffers(self) -> None:
        """Restore deterministic buffers omitted by HF meta-device checkpoint loading."""
        encoder = self.encoder
        device = next(self.parameters()).device
        coordinates, finger_ids, slot_ids, mask = sequence_position_buffers(
            prefix_length=encoder.prefix_length,
            frame_size=encoder.frame_size,
            patch_size=encoder.patch_size,
            history_size=encoder.history_size,
            time_stride=encoder.time_stride,
            canonicalize_right=encoder.canonicalize_right,
        )
        encoder.coordinates = coordinates.to(device)
        encoder.finger_ids = finger_ids.to(device)
        encoder.slot_ids = slot_ids.to(device)
        encoder.causal_mask = mask.to(device)
        for attention in encoder._attention_layers():
            attention_device = next(attention.parameters()).device
            attention.coordinates = coordinates.to(attention_device)
            attention.causal_mask = mask.to(attention_device)

    def train(self, mode: bool = True) -> "FrozenTactileEncoder":
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.ndim != 6 or tuple(pixels.shape[1:]) != (2, 8, 3, 42, 42):
            raise ValueError("tactile_pixel_values must be (B,2,8,3,42,42)")
        if not torch.is_floating_point(pixels):
            raise ValueError(
                "tactile_pixel_values must be finite floating-point BGR [0,1]"
            )
        # Bounds also detect NaN/Inf. Transfer both scalars together instead of
        # synchronizing the device for three separate Python conditionals.
        lower, upper = torch.stack(torch.aminmax(pixels)).cpu().tolist()
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError(
                "tactile_pixel_values must be finite floating-point BGR [0,1]"
            )
        if lower < 0 or upper > 1:
            raise ValueError("tactile_pixel_values must be in [0,1]")
        parameter = next(self.parameters())
        pixels = pixels.to(device=parameter.device, dtype=parameter.dtype)
        return self.encoder.forward_cls(pixels)

    def load_world_model(self, path: str | Path) -> dict:
        state = torch.load(path, map_location="cpu", weights_only=True)
        prefix = "jepa.encoder."
        selected = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if not selected:
            raise ValueError("Checkpoint has no jepa.encoder parameters")
        self.encoder.load_state_dict(selected, strict=True)
        self.requires_grad_(False)
        self.eval()
        return {
            "loaded_keys": sorted(selected),
            "excluded_keys": sorted(key for key in state if not key.startswith(prefix)),
        }
