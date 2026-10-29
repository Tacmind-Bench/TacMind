"""Hydra train loop for DualEncoderJEPA (vendor SIGReg + frozen DINOv3 context).

Does not edit gitignored ``third_party/le-wm/train.py``. Mosaic ``pixels``
stay BGR ``[0, 1]`` (no ImageNet). Agentview uses RGB ImageNet. Skips z-score
on both image columns.
"""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

from tacbenchhub.data.lewm_current_vision import maybe_wrap_current_vision_dataset
from tacbenchhub.data.lewm_visuo_tactile_dataset import (
    ensure_force_column,
    ensure_pixels_hardlink,
)
from tacbenchhub.models.lewm.mosaic_rest import maybe_wrap_rest_sub, parse_mosaic_rest_sub
from tacbenchhub.models.lewm.multi_source_action import SOURCE_NAME_TO_ID
from tacbenchhub.models.lewm.tactile_sequence import (
    MosaicToTactileSequence,
    parse_tactile_layout,
)
from tacbenchhub.models.lewm.train_entry import (
    LEWM,
    prepare_environ,
    require_vendor_clone,
    visuo_jepa_argv,
)
from tacbenchhub.models.lewm.visuo_jepa import (
    DEFAULT_EMBED_DIM,
    MosaicBgrToFloat,
    VisionBgrToRgbImagenet,
    align_cells_to_pred,
    align_force_to_pred,
    current_cell_targets,
    current_pad_temporal_energy,
    force_feature_dim,
    parse_drop_p,
    parse_force_context,
    parse_force_pred,
    parse_force_repr,
    parse_missing_action,
    parse_missing_vision,
    parse_mosaic_layout,
    parse_on_off,
    parse_proprio_context,
    parse_vision_context,
    parse_vision_force_gate,
    parse_vision_history,
    parse_vision_views,
    pred_action_hinge,
    pred_infonce,
    resolve_vision_context,
    vision_keys_for_views,
)

IMAGE_COLUMNS = frozenset({"pixels", "vision", "wrist"})
SKIP_NORMALIZER = IMAGE_COLUMNS | frozenset(
    {"contact", "task_id", "source_id", "has_action", "has_vision"}
)
PRED_MASKS = ("none", "contact", "last", "energy")
PRED_MASK_ALIASES = {
    "none": "none",
    "false": "none",
    "off": "none",
    "contact": "contact",
    "last": "last",
    "energy": "energy",
    "gel": "energy",
    "current_energy": "energy",
    "tactile_energy": "energy",
}


def configure_torch_performance(
    torch_module: Any,
    *,
    float32_matmul_precision: str | None = None,
    allow_tf32: bool | None = None,
) -> None:
    """Apply opt-in CUDA math settings without changing model/data semantics.

    The default is intentionally a no-op.  ``high``/TF32 only affect FP32
    matrix multiplications, while the LeWM runs use bf16 mixed precision for
    the main path.  Keeping this opt-in makes the benchmark able to compare
    the exact vendor defaults before enabling it for production.
    """
    if float32_matmul_precision is not None:
        precision = str(float32_matmul_precision).strip().lower()
        if precision not in {"highest", "high", "medium"}:
            raise ValueError(
                "float32_matmul_precision must be highest, high, or medium, "
                f"got {float32_matmul_precision!r}"
            )
        torch_module.set_float32_matmul_precision(precision)
    if allow_tf32 is not None and getattr(torch_module, "cuda", None) is not None:
        enabled = bool(allow_tf32)
        backends = getattr(torch_module, "backends", None)
        cuda_backend = getattr(backends, "cuda", None) if backends is not None else None
        cudnn_backend = getattr(backends, "cudnn", None) if backends is not None else None
        if cuda_backend is not None and hasattr(cuda_backend, "matmul"):
            cuda_backend.matmul.allow_tf32 = enabled
        if cudnn_backend is not None and hasattr(cudnn_backend, "allow_tf32"):
            cudnn_backend.allow_tf32 = enabled


def source_mix_clip_weights(
    h5_path: Path,
    clip_indices: Any,
    *,
    mix: tuple[float, ...] = (1.0, 0.5, 0.5),
) -> torch.Tensor:
    """Per-clip sample weights so batch mix ≈ ``source_mix_weights``.

    ``mix`` is relative mass per source, in ``SOURCE_NAMES`` order.  The
    legacy 3-tuple (tacbench:touchhd:daimon) is zero-padded for the ftp1
    sources.  Each clip inherits the ``source_id`` of its first frame;
    weight = target_frac[s] / n_clips[s].
    """
    import h5py
    import numpy as np
    import torch

    with h5py.File(h5_path, "r") as handle:
        if "source_id" not in handle:
            return torch.ones(len(clip_indices), dtype=torch.double)
        source = np.asarray(handle["source_id"][:], dtype=np.int64)
    starts = []
    for clip in clip_indices:
        if isinstance(clip, tuple | list):
            starts.append(int(clip[0]))
        else:
            starts.append(int(np.asarray(clip).reshape(-1)[0]))
    starts_arr = np.asarray(starts, dtype=np.int64)
    sid = source[np.clip(starts_arr, 0, source.shape[0] - 1)]
    mix_arr = np.asarray(mix, dtype=np.float64)
    if mix_arr.ndim != 1 or mix_arr.shape[0] < 3 or float(mix_arr.sum()) <= 0:
        raise ValueError(
            f"mix must be >=3 positive masses (SOURCE_NAMES order), got {mix}"
        )
    if mix_arr.shape[0] < len(SOURCE_NAME_TO_ID):
        mix_arr = np.pad(mix_arr, (0, len(SOURCE_NAME_TO_ID) - mix_arr.shape[0]))
    target = mix_arr / mix_arr.sum()
    weights = np.zeros(len(sid), dtype=np.float64)
    for s in range(len(mix_arr)):
        mask = sid == s
        count = int(mask.sum())
        if count == 0:
            continue
        weights[mask] = target[s] / count
    weights = np.maximum(weights, 1e-12)
    return torch.as_tensor(weights, dtype=torch.double)


def _active_source_ids_from_h5(cfg: Any, h5_path: Any) -> set[int]:
    """Return the set of source ids actually present in the dataset.

    Falls back to all registered sources when the file or the ``source_id``
    column is missing, so mixed training is unaffected.  The legacy
    three-source fallback (0/1/2) remains for older three-source H5s.
    """
    all_ids = set(SOURCE_NAME_TO_ID.values())
    legacy_ids = {0, 1, 2}
    if not h5_path:
        return legacy_ids
    try:
        h5_file = Path(str(h5_path)).expanduser()
        if not h5_file.is_file():
            return legacy_ids
        import h5py
        import numpy as np

        with h5py.File(h5_file, "r") as handle:
            if "source_id" not in handle:
                return legacy_ids
            source = np.asarray(handle["source_id"][:], dtype=np.int64)
        return {int(s) for s in np.unique(source).tolist()} & all_ids
    except Exception:
        return legacy_ids


def _parse_source_mix(value: Any) -> tuple[float, ...] | None:
    """Parse ``source_mix_weights`` as one weight per registered source.

    Accepts 3 numbers (legacy tacbench/touchhd/daimon order) or N numbers
    matching ``SOURCE_NAMES`` order (tacbench, touchhd, daimon,
    ftp1_rdp, ftp1_vla_touch, ftp1_qingloong).  A 3-number spec is
    zero-padded for the ftp1 sources so old configs keep working.
    """
    if value is None or value is False:
        return None
    if isinstance(value, str):
        parts = [float(p) for p in value.replace(":", ",").split(",") if p.strip()]
    else:
        parts = [float(p) for p in value]
    n_registered = len(SOURCE_NAME_TO_ID)
    if len(parts) == 3 and n_registered > 3:
        parts = parts + [0.0] * (n_registered - 3)
    if len(parts) != n_registered:
        raise ValueError(
            f"source_mix_weights need 3 or {n_registered} numbers, got {value!r}"
        )
    return tuple(parts)


def parse_pred_mask(value: str | None) -> str:
    if value is None or value is False:
        return "none"
    key = str(value).strip().lower()
    if key not in PRED_MASK_ALIASES:
        raise ValueError(f"pred_mask must be one of {PRED_MASKS}, got {value!r}")
    return PRED_MASK_ALIASES[key]


def select_visuo_jepa_keys(
    keys: list[str],
    views: tuple[str, ...],
    pred_mask: str = "none",
    *,
    vision_residual: bool = True,
    vision_context: str | bool | None = None,
    force_context: str = "off",
    force_pred: bool = False,
    force_repr: str = "raw",
    vision_force_gate: bool = False,
    proprio_context: str = "off",
) -> list[str]:
    """Keep action/proprio; load cameras ``vision_views`` needs; contact if masked.

    Drop camera columns only when vision is fully off so encode does not
    run frozen DINOv3. ``vision_context=action`` still needs cameras even
    if the bool ``vision_residual`` alias is false. ``force_context``,
    ``force_pred``, and ``vision_force_gate`` all load the 6-D
    rest-subtracted wrench. ``force_repr=magdir`` only changes packing;
    it still needs the same column. ``proprio_context=action`` keeps the
    already-loaded 8-D state.
    """
    selected = list(keys)
    uses_vision = resolve_vision_context(vision_context, vision_residual) in (
        "residual",
        "action",
    )
    if uses_vision:
        vision_keys = vision_keys_for_views(views)
        for key in vision_keys:
            if key not in selected:
                selected.append(key)
        drop = {"vision", "wrist"} - set(vision_keys)
        selected = [key for key in selected if key not in drop]
    else:
        selected = [key for key in selected if key not in {"vision", "wrist"}]
    if parse_pred_mask(pred_mask) == "contact" and "contact" not in selected:
        selected.append("contact")
    need_force = (
        parse_force_context(force_context) != "off"
        or parse_force_pred(force_pred)
        or parse_force_repr(force_repr) == "magdir"
        or parse_vision_force_gate(vision_force_gate)
    )
    if need_force and "force" not in selected:
        selected.append("force")
    if parse_proprio_context(proprio_context) == "action" and "proprio" not in selected:
        selected.append("proprio")
    return selected


def action_encoder_input_dim(
    dataset: Any,
    frameskip: int,
    force_context: str = "off",
    force_repr: str = "raw",
    proprio_context: str = "off",
    *,
    vision_context: str | bool | None = None,
    embed_dim: int = DEFAULT_EMBED_DIM,
) -> int:
    """SWM stacks dense actions to ``frameskip * action_dim``; force is strided.

    HDF5 ``action`` is the unstrided window reshaped to
    ``(num_steps, frameskip * dim)``. ``force`` is sampled like pixels
    (``data[::frameskip]``), so concat adds 6 (raw) or 8 (magdir), not
    ``frameskip * 6``. ``proprio`` is cached like action, so WM-23 concat
    adds ``frameskip * proprio_dim``. Vision CLS is sampled like pixels,
    so WM-31 ``vision_context=action`` adds ``embed_dim`` (192), not
    ``frameskip * embed_dim``.
    """
    skip = int(frameskip)
    dim = skip * int(dataset.get_dim("action"))
    if parse_force_context(force_context) == "action":
        dim += force_feature_dim(force_repr)
    if parse_proprio_context(proprio_context) == "action":
        try:
            proprio_dim = int(dataset.get_dim("proprio"))
        except (KeyError, TypeError, ValueError):
            proprio_dim = 8
        dim += skip * proprio_dim
    if parse_vision_context(vision_context) == "action":
        dim += int(embed_dim)
    return dim


def _align_step_weights(
    weights: Any, per_step: Any, n_preds: int, *, kind: str = "weights"
) -> Any:
    """Align a ``(B, T)`` series to JEPA pred steps."""
    import torch

    aligned = torch.as_tensor(weights, device=per_step.device, dtype=per_step.dtype)
    if aligned.ndim == 3 and aligned.size(-1) == 1:
        aligned = aligned.squeeze(-1)
    if aligned.ndim == 1:
        if int(per_step.size(1)) != 1:
            raise ValueError(
                f"{kind} (B,) cannot score a {int(per_step.size(1))}-step "
                "window; need (B, T) aligned to the JEPA window"
            )
        return aligned.unsqueeze(1)
    if aligned.ndim != 2:
        raise ValueError(f"expected {kind} (B, T), got {tuple(aligned.shape)}")
    start = max(int(n_preds), 0)
    if aligned.size(1) == per_step.size(1) + start:
        return aligned[:, start:]
    if aligned.size(1) == per_step.size(1):
        return aligned
    raise ValueError(
        f"{kind} length {int(aligned.size(1))} != pred steps "
        f"{int(per_step.size(1))} (n_preds={n_preds})"
    )


def masked_pred_loss(
    pred_emb: Any,
    tgt_emb: Any,
    *,
    mask: str,
    contact: Any | None = None,
    energy: Any | None = None,
    n_preds: int = 1,
) -> tuple[Any, Any, Any]:
    """Return unmasked mean, train mean, and last-step mean.

    ``last`` keeps the final JEPA window step (next-state). ``contact`` keeps
    target frames whose rest-subtracted force is on. ``energy`` is a soft
    weight from current L7|R7 temporal change (WM-17): rest still trains,
    contact/motion is upweighted. SIGReg is never masked.
    All-zero contact in a batch falls back to the unmasked mean. Logged
    ``pred_loss`` is always unmasked so arms on the same HDF5 stay comparable.
    ``pred_loss_last`` is the next-state term, comparable across
    ``history_size``.
    """
    chosen = parse_pred_mask(mask)
    sqerr = (pred_emb - tgt_emb).pow(2)
    per_step = sqerr.mean(dim=-1)
    unmasked = sqerr.mean()
    last = per_step[:, -1].mean()
    if chosen == "none":
        return unmasked, unmasked, last
    if chosen == "last":
        return unmasked, last, last
    if chosen == "energy":
        if energy is None:
            raise KeyError("pred_mask=energy requires current-pad temporal energy")
        weights = _align_step_weights(energy, per_step, n_preds).clamp_min(0.0)
        scale = weights.mean().clamp_min(1e-8)
        weights = 1.0 + weights / scale
        return unmasked, (per_step * weights).mean() / weights.mean(), last
    if contact is None:
        raise KeyError("pred_mask=contact requires batch['contact']")
    keep = _align_step_weights(contact, per_step, n_preds) > 0.5
    if not bool(keep.any()):
        return unmasked, unmasked, last
    return unmasked, per_step[keep].mean(), last


def _sample_keep_from_output(output: Any, key: str) -> Any | None:
    avail = output.get(key)
    if avail is None:
        return None
    import torch

    tensor = torch.as_tensor(avail).reshape(-1)
    return tensor > 0.5


def apply_sample_keep_pred_loss(
    pred_emb: Any,
    tgt_emb: Any,
    trained: Any,
    keep: Any | None,
) -> Any:
    """Restrict train pred_loss to windows that still have the action."""
    if keep is None:
        return trained
    import torch

    mask = torch.as_tensor(keep, device=pred_emb.device)
    if mask.ndim != 1:
        mask = mask.reshape(-1)
    if mask.numel() != int(pred_emb.size(0)):
        raise ValueError(
            f"available_action {tuple(mask.shape)} != batch {int(pred_emb.size(0))}"
        )
    if bool(mask.all()):
        return trained
    if not bool(mask.any()):
        return pred_emb.new_zeros(())
    per_step = (pred_emb - tgt_emb).pow(2).mean(dim=-1)
    return per_step[mask].mean()


def _hinge_warmup_scale(*, batch_idx: int | None, warmup: int) -> float:
    """Linear hinge ramp: 0 before ``warmup`` optimizer steps, 1 after.

    A from-scratch predictor has zero action sensitivity at step 0
    (adaLN-zero init), so the hinge is maximally active exactly when the
    predictor cannot yet distinguish actions — the cheapest escape is a
    high-gain noise channel (measured on the margin=8 A/B arm,
    2026-09-15).  Plain MSE must first build a meaningful
    context->future mapping; ``warmup <= 0`` disables the ramp.
    """
    if int(warmup) <= 0:
        return 1.0
    idx = 0 if batch_idx is None else int(batch_idx)
    return min(1.0, max(0.0, float(idx) / float(warmup)))


def _action_hinge_loss(
    model: Any,
    batch: dict,
    output: dict,
    ctx_emb: Any,
    pred_emb: Any,
    mse_tgt: Any,
    *,
    ctx_len: int,
    n_preds: int,
    stopgrad: bool,
    margin: float,
) -> Any | None:
    """Train-time ``delta_shuffle`` hinge over rows that have an action.

    Re-encodes the context actions with a within-source shuffle so the
    predictor receives physically inconsistent actions, then requires the
    resulting prediction to be at least ``1 + margin`` times worse than
    the true-action prediction.  Two anti-shortcut guards:

    * **Zero-action anchor** — a predictor can satisfy the shuffle hinge
      by turning the action path into a high-gain noise channel (any
      action perturbation flings the prediction away, zero action stays
      clean).  The eval metric flags exactly that signature as
      ``delta_zero_action < 0``.  Here the zero-action prediction is
      required to be as wrong as the shuffled one, so the only way to
      satisfy both is for *specific action values* to move the prediction
      toward the true future.
    * Rows without an action (``available_action`` 0) are excluded; if no
      row qualifies the loss is ``None`` so the step contributes nothing.
    """
    import torch

    available = output.get("available_action")
    if available is None:
        return None
    keep = torch.as_tensor(available, device=pred_emb.device).reshape(-1) > 0.5
    if not bool(keep.any()):
        return None
    act_emb = output["act_emb"]
    ctx_act = act_emb[:, :ctx_len]
    batch_size = int(ctx_act.size(0))
    if int(ctx_act.size(0)) != int(keep.numel()):
        raise ValueError(
            f"available_action {int(keep.numel())} != act_emb batch "
            f"{int(ctx_act.size(0))}"
        )
    # Step-varying permutation: draw the seed from the global CPU RNG so the
    # pairing changes every step (a constant seed would reuse one permutation
    # pattern for every equally-sized source group).
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(torch.randint(0, 2**31 - 1, (1,)).item()))
    shuffled = ctx_act.clone()
    # SWM windows per-frame meta columns to (B, T); reduce to one id per
    # sample exactly like ``DualEncoderJEPA._per_sample_flag``.  Datasets
    # without a ``source_id`` column (single-source fine-tune H5) fall back
    # to one global permutation over kept rows.
    raw_sid = batch.get("source_id")
    sid_t = torch.as_tensor(raw_sid) if raw_sid is not None else None
    if sid_t is not None and sid_t.numel() > 0:
        if sid_t.ndim >= 2:
            sid_t = sid_t.reshape(batch_size, -1)[:, -1]
        else:
            sid_t = sid_t.reshape(-1)
            if int(sid_t.numel()) != batch_size and batch_size > 0 and int(
                sid_t.numel()
            ) % batch_size == 0:
                sid_t = sid_t.view(batch_size, -1)[:, -1]
    if sid_t is not None and sid_t.numel() == batch_size:
        sid = sid_t.to(ctx_act.device)
        for source in sid.unique().tolist():
            members = (sid == source) & keep
            idx = members.nonzero(as_tuple=False).reshape(-1)
            if int(idx.numel()) >= 2:
                perm = torch.randperm(int(idx.numel()), generator=generator).to(
                    idx.device
                )
                shuffled[idx] = ctx_act[idx[perm]]
    else:
        # No usable source ids: one global permutation over kept rows.
        idx = keep.nonzero(as_tuple=False).reshape(-1)
        if int(idx.numel()) < 2:
            return None
        perm = torch.randperm(int(idx.numel()), generator=generator).to(idx.device)
        shuffled[idx] = ctx_act[idx[perm]]
    # Rows whose action embedding did not actually change (fixed points of
    # the permutation, singleton groups, identical embeddings) must not be
    # scored: their shuffled prediction equals the true one and would only
    # dilute the hinge.
    changed = (shuffled - ctx_act).abs().amax(dim=(1, 2)) > 0
    scored = keep & changed
    if not bool(scored.any()):
        return None
    # Only the conditioning side changes.  ``ctx_emb`` is detached so the
    # hinge cannot push the vision/tactile encoder to degrade context
    # features; the target is detached so the encoder's future embedding is
    # never dragged toward the garbage shuffled prediction.  The shuffled
    # actions themselves are also detached: the hinge must not receive
    # adversarial gradients that repel *any* action embedding from the
    # others (that would corrupt the action embedding space).
    pred_shuffled = model.predict(ctx_emb.detach(), shuffled.detach())
    loss = pred_action_hinge(
        pred_emb[scored],
        pred_shuffled[scored],
        mse_tgt[scored].detach(),
        margin=float(margin),
    )
    return loss


def visuo_jepa_forward(self, batch, stage, cfg):
    """Vendor lejepa_forward plus optional pred_loss mask. Does not edit le-wm."""
    import torch

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight
    mask = parse_pred_mask(getattr(cfg, "pred_mask", None))

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    if "force" in batch:
        batch["force"] = torch.nan_to_num(batch["force"], 0.0)
    if "proprio" in batch:
        batch["proprio"] = torch.nan_to_num(batch["proprio"], 0.0)
    output = self.model.encode(batch)
    emb = output["emb"]
    act_emb = output["act_emb"]
    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:]
    pred_emb = self.model.predict(ctx_emb, ctx_act)
    # Logged pred_loss stays unmasked MSE. Aux terms only enter ``loss``.
    stopgrad = parse_on_off(
        getattr(self.model, "pred_stopgrad", getattr(cfg, "pred_stopgrad", False)),
        name="pred_stopgrad",
    )
    mse_tgt = tgt_emb.detach() if stopgrad else tgt_emb
    layout = parse_mosaic_layout(getattr(self.model, "mosaic_layout", "full"))
    energy = None
    if mask == "energy":
        if "pixels" not in batch:
            raise KeyError("pred_mask=energy requires batch['pixels']")
        energy = current_pad_temporal_energy(batch["pixels"], layout=layout)
    unmasked, trained, last = masked_pred_loss(
        pred_emb,
        mse_tgt,
        mask=mask,
        contact=batch.get("contact"),
        energy=energy,
        n_preds=int(n_preds),
    )
    missing_action = parse_missing_action(
        getattr(self.model, "missing_action", "skip_pred")
    )
    if missing_action == "skip_pred":
        trained = apply_sample_keep_pred_loss(
            pred_emb,
            mse_tgt,
            trained,
            _sample_keep_from_output(output, "available_action"),
        )
    if getattr(self.model, "pred_action_hinge", False) and self.training:
        warmup = int(getattr(self.model, "pred_action_hinge_warmup", 0) or 0)
        if warmup > 0:
            # spt.Module injects the per-epoch batch_idx; global_step is
            # not exposed to forward, so accumulate across epochs through
            # the epoch counter when present (epoch * its own length would
            # need trainer state we do not have here).
            batch_idx = batch.get("batch_idx")
            try:
                batch_idx = None if batch_idx is None else int(batch_idx)
            except (TypeError, ValueError):
                batch_idx = None
            hinge_scale = _hinge_warmup_scale(batch_idx=batch_idx, warmup=warmup)
        else:
            hinge_scale = 1.0
        if hinge_scale > 0.0:
            hinge = _action_hinge_loss(
                self.model,
                batch,
                output,
                ctx_emb,
                pred_emb,
                mse_tgt,
                ctx_len=int(ctx_len),
                n_preds=int(n_preds),
                stopgrad=stopgrad,
                margin=getattr(self.model, "pred_action_hinge_margin", 3.0),
            )
            if hinge is not None:
                hinge_w = float(
                    getattr(self.model, "pred_action_hinge_weight", 1.0) or 0.0
                )
                output["pred_action_hinge_loss"] = hinge
                trained = trained + hinge_w * hinge_scale * hinge
    output["pred_loss"] = unmasked
    output["pred_loss_train"] = trained
    output["pred_loss_last"] = last
    if parse_on_off(
        getattr(self.model, "pred_nce", getattr(cfg, "pred_nce", False)),
        name="pred_nce",
    ):
        nce = pred_infonce(
            pred_emb,
            tgt_emb,
            temperature=float(getattr(self.model, "pred_nce_temp", 0.1) or 0.1),
        )
        output["pred_nce_loss"] = nce
        nce_w = float(getattr(self.model, "pred_nce_weight", 0.0) or 0.0)
        trained = trained + nce_w * nce
    recon_w = float(getattr(self.model, "current_recon_weight", 0.0) or 0.0)
    if parse_on_off(
        getattr(self.model, "current_recon", False), name="current_recon"
    ):
        if "pixels" not in batch:
            raise KeyError("current_recon=true requires batch['pixels']")
        target = current_cell_targets(batch["pixels"], layout=layout).to(
            device=emb.device, dtype=emb.dtype
        )
        recon = (self.model.decode_current_cells(emb) - target).pow(2).mean()
        output["current_recon_loss"] = recon
        trained = trained + recon_w * recon
    next_w = float(getattr(self.model, "next_recon_weight", 0.0) or 0.0)
    if parse_on_off(
        getattr(self.model, "next_recon", False), name="next_recon"
    ):
        if "pixels" not in batch:
            raise KeyError("next_recon=true requires batch['pixels']")
        next_target = align_cells_to_pred(
            current_cell_targets(batch["pixels"], layout=layout),
            pred_emb,
            int(n_preds),
        )
        next_loss = (self.model.decode_next_cells(pred_emb) - next_target).pow(2).mean()
        output["next_recon_loss"] = next_loss
        trained = trained + next_w * next_loss
    if parse_on_off(
        getattr(self.model, "force_pred", False), name="force_pred"
    ):
        if "force" not in batch:
            raise KeyError("force_pred=true requires batch['force']")
        target = align_force_to_pred(batch["force"], pred_emb, int(n_preds))
        force_loss = (self.model.decode_force(pred_emb) - target).pow(2).mean()
        output["force_pred_loss"] = force_loss
        force_w = float(getattr(self.model, "force_pred_weight", 0.0) or 0.0)
        trained = trained + force_w * force_loss
    if stage != "train":
        drop_v = parse_drop_p(
            getattr(self.model, "drop_vision_p", 0.0), name="drop_vision_p"
        )
        drop_a = parse_drop_p(
            getattr(self.model, "drop_action_p", 0.0), name="drop_action_p"
        )
        if drop_v > 0:
            dropped = self.model.encode_with_forced_drop(
                batch, vision_p=1.0, action_p=0.0
            )
            pred_d = self.model.predict(
                dropped["emb"][:, :ctx_len], dropped["act_emb"][:, :ctx_len]
            )
            tgt_d = dropped["emb"][:, n_preds:]
            d_unmasked, _, d_last = masked_pred_loss(
                pred_d,
                tgt_d.detach() if stopgrad else tgt_d,
                mask="none",
                n_preds=int(n_preds),
            )
            output["pred_loss_drop_vision"] = d_unmasked
            output["pred_loss_last_drop_vision"] = d_last
        if drop_a > 0 and missing_action != "skip_pred":
            dropped = self.model.encode_with_forced_drop(
                batch, vision_p=0.0, action_p=1.0
            )
            pred_d = self.model.predict(
                dropped["emb"][:, :ctx_len], dropped["act_emb"][:, :ctx_len]
            )
            tgt_d = dropped["emb"][:, n_preds:]
            d_unmasked, _, d_last = masked_pred_loss(
                pred_d,
                tgt_d.detach() if stopgrad else tgt_d,
                mask="none",
                n_preds=int(n_preds),
            )
            output["pred_loss_drop_action"] = d_unmasked
            output["pred_loss_last_drop_action"] = d_last
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = trained + lambd * output["sigreg_loss"]
    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


def trainable_module_pattern(
    *,
    vision_residual: bool = True,
    vision_context: str | bool | None = None,
    force_context: str = "off",
    current_recon: bool = False,
    force_pred: bool = False,
    next_recon: bool = False,
    missing_vision: str | bool | None = "skip",
    missing_action: str | bool | None = "skip_pred",
) -> str:
    """spt.Module matches this against ``named_modules()``.

    Exclude frozen DINOv3 so AdamW does not allocate moments for
    ``model.vision_encoder``. ``vision_proj`` trains for residual or
    action-site vision. ``force_proj`` is only trained for
    ``force_context=residual``. ``current_cell_head`` is only trained
    for WM-18 ``current_recon``. ``force_head`` is only trained
    for WM-19/20 ``force_pred``. ``next_cell_head`` is only trained
    for WM-26 ``next_recon``. ``missing_tokens`` is only trained when
    ``missing_vision`` or ``missing_action`` is ``null``.
    """
    modules = ["jepa"]
    uses_vision = resolve_vision_context(vision_context, vision_residual) in (
        "residual",
        "action",
    )
    if uses_vision:
        modules.extend(["vision_proj", "vision_fuse_proj"])
    if parse_force_context(force_context) == "residual":
        modules.append("force_proj")
    if current_recon:
        modules.append("current_cell_head")
    if next_recon:
        modules.append("next_cell_head")
    if parse_force_pred(force_pred):
        modules.append("force_head")
    uses_null = (
        parse_missing_vision(missing_vision) == "null"
        or parse_missing_action(missing_action) == "null"
    )
    if uses_null:
        modules.append("missing_tokens")
    return rf"^model\.({'|'.join(modules)})(\.|$)"


TRAINABLE_MODULE_PATTERN = trainable_module_pattern()


def set_data_loader_worker_threads(worker_id: int, *, threads: int) -> None:
    """Dataloader ``worker_init_fn`` clamping per-worker torch threads.

    ``threads <= 0`` falls back to 1 so a misconfigured YAML cannot spawn
    workers that each try to use every core.
    """
    import torch

    torch.set_num_threads(max(int(threads), 1))


def _zscore_column(dataset: Any, column: str, *, std_floor: float = 1e-6) -> Any:
    """Z-score a vector column, clamping std so rest-subtracted force cannot /0."""
    import numpy as np
    import stable_pretraining as spt
    import torch
    from utils import ZScoreNormalizer

    data = torch.from_numpy(np.array(dataset.get_col_data(column))).float()
    if data.ndim == 1:
        data = data.unsqueeze(-1)
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone().clamp_min(float(std_floor))
    return spt.data.transforms.WrapTorchTransform(
        ZScoreNormalizer(mean, std), source=column, target=column
    )


def episode_task_ids(h5_path: Path) -> list[int]:
    """One task id per episode from SWM ``task_id[ep_offset]``."""
    import h5py

    with h5py.File(h5_path, "r") as handle:
        n_ep = int(handle["ep_len"].shape[0])
        if "task_id" not in handle or "ep_offset" not in handle:
            return [0] * n_ep
        offsets = handle["ep_offset"][:]
        return [int(value) for value in handle["task_id"][:][offsets]]


def stratified_episode_split(
    task_per_episode: list[int],
    *,
    train_split: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """90/10 episode split inside each task. No window leakage across splits."""
    import numpy as np

    by_task: dict[int, list[int]] = {}
    for episode, task in enumerate(task_per_episode):
        by_task.setdefault(int(task), []).append(int(episode))
    rng = np.random.default_rng(int(seed))
    train: list[int] = []
    val: list[int] = []
    for task in sorted(by_task):
        order = np.asarray(by_task[task], dtype=np.int64)
        rng.shuffle(order)
        n_ep = int(order.size)
        if n_ep == 1:
            train.append(int(order[0]))
            continue
        n_train = int(round(n_ep * float(train_split)))
        n_train = min(max(n_train, 1), n_ep - 1)
        train.extend(int(value) for value in order[:n_train])
        val.extend(int(value) for value in order[n_train:])
    if not train or not val:
        raise ValueError("stratified episode split needs at least one train and val episode")
    return sorted(train), sorted(val)


def subset_clips_by_episode(dataset: Any, episode_ids: set[int]) -> Any:
    """Keep SWM windows whose episode is in ``episode_ids``."""
    import torch

    clips = getattr(dataset, "clip_indices", None)
    if clips is None:
        raise TypeError("dataset has no clip_indices; cannot split by episode")
    indices = [
        index for index, (episode, _start) in enumerate(clips) if int(episode) in episode_ids
    ]
    if not indices:
        raise ValueError("episode split produced an empty subset")
    return torch.utils.data.Subset(dataset, indices)


def _train(cfg: Any) -> None:
    import hydra
    import lightning as pl
    import stable_pretraining as spt
    import stable_worldmodel as swm
    import torch
    from lightning.pytorch.loggers import WandbLogger
    from module import SIGReg
    from omegaconf import OmegaConf, open_dict
    from utils import SaveCkptCallback, get_column_normalizer

    configure_torch_performance(
        torch,
        float32_matmul_precision=OmegaConf.select(
            cfg, "performance.float32_matmul_precision", default=None
        ),
        allow_tf32=(
            None
            if OmegaConf.select(cfg, "performance.allow_tf32", default=None) is None
            else parse_on_off(
                OmegaConf.select(cfg, "performance.allow_tf32"), name="allow_tf32"
            )
        ),
    )
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    current_vision_only = parse_on_off(
        dataset_cfg.pop("current_vision_only", False), name="current_vision_only"
    )
    views = parse_vision_views(
        OmegaConf.select(cfg, "model.vision_views", default="agentview")
    )
    layout = parse_mosaic_layout(
        OmegaConf.select(cfg, "model.mosaic_layout", default="full")
    )
    tactile_layout = parse_tactile_layout(
        OmegaConf.select(cfg, "model.tactile_layout", default="mosaic")
    )
    pred_mask = parse_pred_mask(OmegaConf.select(cfg, "pred_mask", default="none"))
    residual = bool(OmegaConf.select(cfg, "model.vision_residual", default=True))
    vision_context = OmegaConf.select(cfg, "model.vision_context", default=None)
    vision_history = parse_vision_history(
        OmegaConf.select(cfg, "model.vision_history", default="all")
    )
    force_context = parse_force_context(
        OmegaConf.select(cfg, "model.force_context", default="off")
    )
    force_pred = parse_force_pred(
        OmegaConf.select(cfg, "model.force_pred", default=False)
    )
    force_repr = parse_force_repr(
        OmegaConf.select(cfg, "model.force_repr", default="raw")
    )
    vision_force_gate = parse_vision_force_gate(
        OmegaConf.select(cfg, "model.vision_force_gate", default=False)
    )
    proprio_context = parse_proprio_context(
        OmegaConf.select(cfg, "model.proprio_context", default="off")
    )
    keys_to_load = select_visuo_jepa_keys(
        list(dataset_cfg.get("keys_to_load") or []),
        views,
        pred_mask=pred_mask,
        vision_residual=residual,
        vision_context=vision_context,
        force_context=force_context,
        force_pred=force_pred,
        force_repr=force_repr,
        vision_force_gate=vision_force_gate,
        proprio_context=proprio_context,
    )
    h5_path = Path(str(dataset_name))
    if h5_path.is_file():
        import h5py

        with h5py.File(h5_path, "r") as handle:
            for meta_key in ("source_id", "has_action", "has_vision"):
                if meta_key in handle and meta_key not in keys_to_load:
                    keys_to_load.append(meta_key)
            # The base config caches proprio for single-source TacBench; the
            # cotrain merges omit that column, so drop it from the cache
            # list when the H5 does not carry it.
            cache_keys = dataset_cfg.get("keys_to_cache")
            if cache_keys is not None:
                dataset_cfg["keys_to_cache"] = [k for k in cache_keys if k in handle]
                if dataset_cfg["keys_to_cache"] != list(cache_keys):
                    print(
                        "[keys_to_cache] dropped missing columns: "
                        f"{[k for k in cache_keys if k not in handle]}",
                        flush=True,
                    )
    dataset_cfg["keys_to_load"] = keys_to_load
    need_force = (
        force_context != "off"
        or force_pred
        or force_repr == "magdir"
        or vision_force_gate
    )
    if need_force:
        cache = list(dataset_cfg.get("keys_to_cache") or [])
        if "force" not in cache:
            cache.append("force")
        dataset_cfg["keys_to_cache"] = cache
    if "pixels" in keys_to_load and h5_path.is_file():
        ensure_pixels_hardlink(h5_path)
    if need_force and h5_path.is_file():
        ensure_force_column(h5_path)
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    dataset = maybe_wrap_current_vision_dataset(
        dataset,
        enabled=current_vision_only and vision_history == "current",
        vision_keys=vision_keys_for_views(views),
        history_size=int(cfg.history_size),
        num_steps=int(dataset_cfg["num_steps"]),
    )
    if tactile_layout == "sequence_3x3":
        tactile_transform = MosaicToTactileSequence(
            frame_size=int(
                OmegaConf.select(cfg, "model.tactile_frame_size", default=42)
            ),
            history_size=int(
                OmegaConf.select(cfg, "model.tactile_history_size", default=8)
            ),
            history_stride=int(
                OmegaConf.select(cfg, "model.tactile_history_stride", default=5)
            ),
        )
    else:
        tactile_transform = MosaicBgrToFloat(img_size=int(cfg.img_size), layout=layout)
    transforms = [
        spt.data.transforms.WrapTorchTransform(
            tactile_transform,
            source="pixels",
            target="pixels",
        ),
    ]
    for camera_key in vision_keys_for_views(views):
        if camera_key not in keys_to_load:
            continue
        transforms.append(
            spt.data.transforms.WrapTorchTransform(
                VisionBgrToRgbImagenet(img_size=int(cfg.img_size)),
                source=camera_key,
                target=camera_key,
            )
        )
    with open_dict(cfg):
        cfg.pred_mask = pred_mask
        cfg.data.dataset.keys_to_load = keys_to_load
        for col in keys_to_load:
            if col in SKIP_NORMALIZER or str(col).startswith("pixels"):
                continue
            if col == "force":
                transforms.append(_zscore_column(dataset, col, std_floor=1e-6))
                continue
            transforms.append(get_column_normalizer(dataset, col, col))
        cfg.model.action_encoder.input_dim = action_encoder_input_dim(
            dataset,
            int(cfg.data.dataset.frameskip),
            force_context,
            force_repr,
            proprio_context,
            vision_context=vision_context,
            embed_dim=int(OmegaConf.select(cfg, "embed_dim", default=DEFAULT_EMBED_DIM)),
        )

    dataset.transform = spt.data.transforms.Compose(*transforms)
    rest_sub = parse_mosaic_rest_sub(
        OmegaConf.select(cfg, "model.mosaic_rest_sub", default="off")
    )
    if h5_path.is_file():
        dataset = maybe_wrap_rest_sub(dataset, h5_path, rest_sub)
    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    if h5_path.is_file() and hasattr(dataset, "clip_indices"):
        train_eps, val_eps = stratified_episode_split(
            episode_task_ids(h5_path),
            train_split=float(cfg.train_split),
            seed=int(cfg.seed),
        )
        train_set = subset_clips_by_episode(dataset, set(train_eps))
        val_set = subset_clips_by_episode(dataset, set(val_eps))
    else:
        train_set, val_set = spt.data.random_split(
            dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
        )
    # PyTorch only accepts prefetch_factor/persistent_workers with worker processes.
    loader_kwargs = dict(cfg.loader)
    worker_threads = loader_kwargs.pop("worker_threads", None)
    if int(loader_kwargs.get("num_workers", 0)) == 0:
        loader_kwargs.pop("prefetch_factor", None)
        loader_kwargs.pop("persistent_workers", None)
    elif worker_threads is not None and int(worker_threads) > 0:
        loader_kwargs["worker_init_fn"] = partial(
            set_data_loader_worker_threads, threads=int(worker_threads)
        )
    source_mix = _parse_source_mix(
        OmegaConf.select(cfg, "source_mix_weights", default=None)
    )
    train_sampler = None
    train_shuffle = True
    if source_mix is not None and h5_path.is_file() and hasattr(train_set, "clip_indices"):
        mix_weights = source_mix_clip_weights(
            h5_path, train_set.clip_indices, mix=source_mix
        )
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=mix_weights,
            num_samples=len(mix_weights),
            replacement=True,
            generator=rnd_gen,
        )
        train_shuffle = False
    train = torch.utils.data.DataLoader(
        train_set,
        **loader_kwargs,
        shuffle=train_shuffle,
        sampler=train_sampler,
        drop_last=True,
        generator=rnd_gen,
    )
    val = torch.utils.data.DataLoader(
        val_set, **loader_kwargs, shuffle=False, drop_last=False
    )

    world_model = hydra.utils.instantiate(cfg.model)
    residual = bool(getattr(world_model, "vision_residual", residual))
    model_vision_context = getattr(
        world_model, "vision_context", resolve_vision_context(vision_context, residual)
    )

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(visuo_jepa_forward, cfg=cfg),
        optim={
            "model_opt": {
                "modules": trainable_module_pattern(
                    vision_residual=residual,
                    vision_context=model_vision_context,
                    force_context=force_context,
                    current_recon=parse_on_off(
                        getattr(world_model, "current_recon", False),
                        name="current_recon",
                    ),
                    force_pred=parse_on_off(
                        getattr(world_model, "force_pred", False),
                        name="force_pred",
                    ),
                    next_recon=parse_on_off(
                        getattr(world_model, "next_recon", False),
                        name="next_recon",
                    ),
                    missing_vision=getattr(world_model, "missing_vision", "skip"),
                    missing_action=getattr(world_model, "missing_action", "skip_pred"),
                ),
                "optimizer": dict(cfg.optimizer),
                "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
                "interval": "epoch",
            }
        },
    )

    init_weights = os.environ.get("TACBENCH_INIT_WEIGHTS", "").strip()
    if init_weights:
        import logging

        weight_path = Path(init_weights).expanduser()
        if not weight_path.is_file():
            raise FileNotFoundError(f"TACBENCH_INIT_WEIGHTS not found: {weight_path}")
        if not weight_path.is_absolute():
            raise ValueError(
                f"TACBENCH_INIT_WEIGHTS must be absolute, got {weight_path}"
            )
        payload = torch.load(weight_path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "state_dict" in payload:
            payload = payload["state_dict"]
        if not isinstance(payload, dict):
            raise TypeError(
                f"TACBENCH_INIT_WEIGHTS must be a state_dict, got {type(payload)}"
            )
        cleaned = {
            (key[6:] if str(key).startswith("model.") else key): value
            for key, value in payload.items()
        }
        incompatible = world_model.model.load_state_dict(cleaned, strict=False)
        logging.info(
            "Loaded TACBENCH_INIT_WEIGHTS=%s missing=%d unexpected=%d",
            weight_path,
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
        )

        # Fine-tuning on a narrower distribution than the init weights was
        # trained on will drift BN running stats within a single epoch,
        # shifting every embedding.  Freeze running stats while the user is
        # only updating weights; they can still be calibrated afterwards by
        # a separate ``calibrate_bn`` pass if the new data is the target
        # distribution.  Controlled by TACBENCH_INIT_FREEZE_BN (default 1).
        freeze_bn = os.environ.get("TACBENCH_INIT_FREEZE_BN", "1").strip() not in (
            "0",
            "false",
            "False",
            "no",
            "off",
        )
        if freeze_bn:
            frozen = 0
            for module in world_model.model.modules():
                if isinstance(module, torch.nn.BatchNorm1d | torch.nn.SyncBatchNorm):
                    module.track_running_stats = True
                    module.running_mean.requires_grad_(False)
                    module.running_var.requires_grad_(False)
                    module.num_batches_tracked.requires_grad_(False)
                    frozen += 1
            logging.info(
                "TACBENCH_INIT_FREEZE_BN froze running stats on %d BN modules",
                frozen,
            )

        # When fine-tuning on a single-source subset, freeze the other
        # source Linears in MultiSourceActionEmbedder and make them
        # identity so they cannot poison the shared Embedder.  Controlled
        # by TACBENCH_INIT_FREEZE_DEAD_SOURCES (default 1) and the
        # dataset's actual ``source_id`` column when available.
        freeze_dead = os.environ.get(
            "TACBENCH_INIT_FREEZE_DEAD_SOURCES", "1"
        ).strip() not in ("0", "false", "False", "no", "off")
        action_encoder = getattr(world_model.model, "action_encoder", None)
        if freeze_dead and hasattr(action_encoder, "set_active_sources"):
            # An explicit TACBENCH_INIT_ACTIVE_SOURCES (e.g. "0" for a
            # TacBench-only fine-tune) wins over the dataset probe, because
            # the fine-tune H5 usually has no ``source_id`` column and the
            # runtime ``forward`` defaults every sample to source 0.
            explicit = os.environ.get("TACBENCH_INIT_ACTIVE_SOURCES", "").strip()
            if explicit:
                active = {int(s) for s in explicit.replace(":", ",").split(",") if s.strip()}
                active_source = "explicit"
            else:
                active = _active_source_ids_from_h5(
                    cfg, h5_path=getattr(cfg.data.dataset, "name", None)
                )
                active_source = "dataset"
            action_encoder.set_active_sources(active)
            logging.info(
                "TACBENCH_INIT_FREEZE_DEAD_SOURCES froze "
                "MultiSourceActionEmbedder to sources=%s (via %s)",
                sorted(active),
                active_source,
            )

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"), run_id)
    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as handle:
        OmegaConf.save(cfg, handle)
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[
            SaveCkptCallback(
                run_name=cfg.output_model_name,
                cfg=cfg.model,
                epoch_interval=int(
                    OmegaConf.select(cfg, "checkpoint_epoch_interval", default=1)
                ),
            )
        ],
        num_sanity_val_steps=1,
        logger=logger,
        # SaveCkptCallback writes STABLEWM ``weights_epoch_*.pt``. Lightning
        # ``last.ckpt`` under ~/.cache/stable-pretraining is a second full copy.
        enable_checkpointing=False,
    )
    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    resume = os.environ.get("TACBENCH_RESUME_CKPT", "").strip()
    weights_only = True
    if resume:
        ckpt_path = Path(resume).expanduser()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"TACBENCH_RESUME_CKPT not found: {ckpt_path}")
        if not ckpt_path.is_absolute():
            raise ValueError(f"TACBENCH_RESUME_CKPT must be absolute, got {ckpt_path}")
        weights_only = False
    elif not ckpt_path.is_file():
        ckpt_path = None
    # Fine-tune from TACBENCH_INIT_WEIGHTS starts a fresh optimizer; do not
    # also pass a Lightning ckpt_path unless the user explicitly resumes.
    if init_weights and not resume:
        ckpt_path = None
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=str(ckpt_path) if ckpt_path is not None else None,
        weights_only=weights_only,
    )
    manager()


def run() -> None:
    import sys

    import hydra

    require_vendor_clone()
    prepare_environ()
    sys.argv = visuo_jepa_argv(sys.argv)
    os.chdir(LEWM)
    hydra.main(version_base=None, config_path="config/train", config_name="lewm")(_train)()
