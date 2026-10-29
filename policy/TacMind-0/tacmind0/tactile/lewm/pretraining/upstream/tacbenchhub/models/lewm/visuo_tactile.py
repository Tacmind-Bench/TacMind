"""One-hop visuo-tactile LeWM: two encoders, fusion, contact head.

Not a vendor JEPA fork and not an official ``success_rate`` policy.
``action -> contact -> tactile`` is a teacher-forced extra input to ``P_t``,
not a causal graph.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_EMBED_DIM = 64
DEFAULT_HIDDEN_DIM = 128
DEFAULT_CONTACT_WEIGHT = 0.1
DEFAULT_PROPRIO_DIM = 8
DEFAULT_ACTION_DIM = 8


def _cosine_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_n = F.normalize(pred, dim=-1, eps=1e-8)
    target_n = F.normalize(target, dim=-1, eps=1e-8)
    return 1.0 - (pred_n * target_n).sum(dim=-1).mean()


class TinyImageEncoder(nn.Module):
    """Small conv CLS encoder. Separate instances for vision and tactile."""

    def __init__(self, embed_dim: int = DEFAULT_EMBED_DIM) -> None:
        super().__init__()
        dim = int(embed_dim)
        self.embed_dim = dim
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, dim, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"expected (B, 3, H, W), got {tuple(image.shape)}")
        pooled = self.net(image.float()).flatten(1)
        return self.norm(self.proj(pooled))


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(output_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VisuoTactileLeWM(nn.Module):
    """``z^v=E_v(V)``, ``z^t=E_t(T)``, ``h=Fusion(z^v,z^t,q)``, one-step heads."""

    def __init__(
        self,
        *,
        embed_dim: int = DEFAULT_EMBED_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        proprio_dim: int = DEFAULT_PROPRIO_DIM,
        action_dim: int = DEFAULT_ACTION_DIM,
        contact_weight: float = DEFAULT_CONTACT_WEIGHT,
        contact_pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        self.contact_weight = float(contact_weight)
        self.contact_pos_weight = (
            None if contact_pos_weight is None else float(contact_pos_weight)
        )
        if self.contact_pos_weight is not None:
            self.register_buffer(
                "contact_pos_weight_tensor",
                torch.tensor([self.contact_pos_weight], dtype=torch.float32),
            )
        self.vision_encoder = TinyImageEncoder(self.embed_dim)
        self.tactile_encoder = TinyImageEncoder(self.embed_dim)
        self.fusion = _MLP(
            self.embed_dim * 2 + self.proprio_dim,
            self.hidden_dim,
            self.embed_dim,
        )
        self.action_embed = nn.Linear(self.action_dim, self.embed_dim)
        fused_action = self.embed_dim * 2
        self.visual_pred = _MLP(fused_action, self.hidden_dim, self.embed_dim)
        self.contact_head = nn.Linear(fused_action, 1)
        self.tactile_pred = _MLP(fused_action + 1, self.hidden_dim, self.embed_dim)

    def encode(
        self,
        vision: torch.Tensor,
        tactile: torch.Tensor,
        proprio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_v = self.vision_encoder(vision)
        z_t = self.tactile_encoder(tactile)
        if proprio.ndim != 2 or proprio.shape[-1] != self.proprio_dim:
            raise ValueError(
                f"expected proprio (B, {self.proprio_dim}), got {tuple(proprio.shape)}"
            )
        h = self.fusion(torch.cat([z_v, z_t, proprio.float()], dim=-1))
        return z_v, z_t, h

    def _fused_action(self, h: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if action.ndim != 2 or action.shape[-1] != self.action_dim:
            raise ValueError(
                f"expected action (B, {self.action_dim}), got {tuple(action.shape)}"
            )
        return torch.cat([h, self.action_embed(action.float())], dim=-1)

    def forward(
        self,
        vision: torch.Tensor,
        tactile: torch.Tensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
        next_contact: torch.Tensor | None = None,
        *,
        teacher_force: bool = True,
    ) -> dict[str, torch.Tensor]:
        z_v, z_t, h = self.encode(vision, tactile, proprio)
        fused = self._fused_action(h, action)
        z_v_hat = self.visual_pred(fused)
        contact_logit = self.contact_head(fused).reshape(h.size(0))
        if teacher_force:
            if next_contact is None:
                raise ValueError("teacher_force requires next_contact")
            contact_in = next_contact.reshape(h.size(0), 1).float()
        else:
            contact_in = torch.sigmoid(contact_logit).reshape(h.size(0), 1)
        z_t_hat = self.tactile_pred(torch.cat([fused, contact_in], dim=-1))
        return {
            "z_v": z_v,
            "z_t": z_t,
            "h": h,
            "z_v_hat": z_v_hat,
            "z_t_hat": z_t_hat,
            "contact_logit": contact_logit,
            "contact_in": contact_in.reshape(h.size(0)),
        }

    def loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        teacher_force: bool = True,
    ) -> dict[str, torch.Tensor]:
        out = self.forward(
            batch["vision"],
            batch["tactile"],
            batch["proprio"],
            batch["action"],
            batch.get("next_contact"),
            teacher_force=teacher_force,
        )
        with torch.no_grad():
            z_v_tgt = self.vision_encoder(batch["next_vision"])
            z_t_tgt = self.tactile_encoder(batch["next_tactile"])
        vision_loss = _cosine_distance(out["z_v_hat"], z_v_tgt)
        tactile_loss = _cosine_distance(out["z_t_hat"], z_t_tgt)
        bce_kwargs: dict[str, torch.Tensor] = {}
        pos_weight = getattr(self, "contact_pos_weight_tensor", None)
        if pos_weight is not None:
            bce_kwargs["pos_weight"] = pos_weight.to(dtype=out["contact_logit"].dtype)
        contact_loss = F.binary_cross_entropy_with_logits(
            out["contact_logit"],
            batch["next_contact"].reshape(out["contact_logit"].shape).float(),
            **bce_kwargs,
        )
        total = vision_loss + tactile_loss + self.contact_weight * contact_loss
        return {
            "loss": total,
            "vision": vision_loss,
            "tactile": tactile_loss,
            "contact": contact_loss,
        }


def train_one_step(
    model: VisuoTactileLeWM,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    teacher_force: bool = True,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad()
    device_type = next(model.parameters()).device.type
    use_amp = scaler is not None and device_type == "cuda"
    with torch.autocast(device_type=device_type, enabled=use_amp):
        losses = model.loss(batch, teacher_force=teacher_force)
    if scaler is None:
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    else:
        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    return {key: float(value.detach()) for key, value in losses.items()}


@torch.inference_mode()
def eval_losses(
    model: VisuoTactileLeWM,
    batch: dict[str, torch.Tensor],
    *,
    teacher_force: bool = False,
) -> dict[str, float]:
    """Diagnostic loss using predicted contact unless ``teacher_force`` is set."""
    was_training = model.training
    model.eval()
    losses = model.loss(batch, teacher_force=teacher_force)
    if was_training:
        model.train()
    return {key: float(value.detach()) for key, value in losses.items()}




def nchw_float(image_bgr: Any) -> torch.Tensor:
    """Convert HWC BGR uint8 (or a batch) to NCHW float in ``[0, 1]``."""
    tensor = torch.as_tensor(image_bgr)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4 or tensor.shape[-1] != 3:
        raise ValueError(f"expected (... H W 3), got {tuple(tensor.shape)}")
    return tensor.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
