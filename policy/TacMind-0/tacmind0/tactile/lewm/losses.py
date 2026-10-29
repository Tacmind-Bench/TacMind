"""Tac-LeWM regularization and action-conditioned prediction objectives."""

import math
from typing import Any

import torch
from torch.nn import functional as F


class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()  # average over projections and time


def pred_infonce(
    pred_emb: Any,
    tgt_emb: Any,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    """In-batch InfoNCE of pred vs stop-grad target, scaled by ``log(N)``.

    Each ``(batch, time)`` step is one query. The matching target step is
    the positive; other steps in the batch (other time, other episode) are
    negatives. Dividing by ``log(N)`` keeps a random match near 1 so λ
    stays comparable to ``force_pred_weight``.
    """
    pred = torch.as_tensor(pred_emb)
    tgt = torch.as_tensor(tgt_emb)
    if pred.shape != tgt.shape or pred.ndim != 3:
        raise ValueError(
            f"pred_infonce expects matching (B, T, D), got "
            f"{tuple(pred.shape)} vs {tuple(tgt.shape)}"
        )
    pred_n = F.normalize(pred.reshape(-1, pred.size(-1)).float(), dim=-1)
    tgt_n = F.normalize(tgt.detach().reshape(-1, tgt.size(-1)).float(), dim=-1)
    n_items = int(pred_n.size(0))
    logits = pred_n @ tgt_n.transpose(0, 1) / max(float(temperature), 1e-6)
    labels = torch.arange(n_items, device=pred_n.device)
    nce = F.cross_entropy(logits, labels)
    return nce / math.log(max(n_items, 2))


def pred_action_hinge(
    pred_true: Any,
    pred_shuffled: Any,
    target: Any,
    *,
    margin: float = 0.05,
) -> torch.Tensor:
    """Penalize predictions that ignore action using a batch-mean cosine gap.

    Inputs are matching ``(B, T, D)`` tensors. The loss is
    ``relu(mean(cos(shuffled, target) - cos(true, target)) + margin)``.
    ``margin`` must lie in ``[0, 2)``. Empty inputs return a scalar zero.
    """
    pred_true = torch.as_tensor(pred_true)
    pred_shuffled = torch.as_tensor(pred_shuffled)
    target = torch.as_tensor(target)
    if pred_true.shape != pred_shuffled.shape or pred_true.shape != target.shape:
        raise ValueError(
            f"pred_action_hinge expects matching (B, T, D), got "
            f"{tuple(pred_true.shape)} / {tuple(pred_shuffled.shape)} / "
            f"{tuple(target.shape)}"
        )
    if float(margin) < 0.0 or float(margin) >= 2.0:
        raise ValueError(f"margin must be in [0, 2), got {margin}")
    if pred_true.numel() == 0:
        return pred_true.new_zeros(())
    dim = pred_true.size(-1)

    def cos(a, b):
        return F.cosine_similarity(
            a.reshape(-1, dim), b.reshape(-1, dim), dim=-1
        ).reshape(a.shape[:-1])

    gap = cos(pred_shuffled, target) - cos(pred_true, target)
    return F.relu(gap.mean() + float(margin))
