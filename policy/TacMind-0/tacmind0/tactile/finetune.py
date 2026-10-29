"""Fine-tune pretrained Tac-LeWM on episode-isolated TacMind-0 marker data.

This is a separate world-model stage. The downstream policy still freezes the
exported tactile encoder. No source repository or online model download is used.
"""

import argparse
import gc
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from tacmind0.data.episodes import (
    compute_action_statistics,
    read_episode_rows,
    split_episodes,
)
from tacmind0.data.sample_index import EpisodeSampleIndex
from tacmind0.data.transforms import LoadImages
from tacmind0.runtime.checkpointing import staged_checkpoint
from tacmind0.runtime.distributed import average_gradients
from tacmind0.tactile.lewm.losses import SIGReg, pred_action_hinge
from tacmind0.tactile.lewm.model import RESOURCE, load_world_model
from tacmind0.tactile.preprocessing import HISTORY, PreparedMarkerCache, history_indices


def freeze_batchnorm_buffers(module: torch.nn.Module) -> int:
    """Keep BatchNorm evaluation statistics fixed during fine-tuning.

    The fine-tune launcher uses manual gradient averaging instead of DDP, so
    BatchNorm buffers would otherwise diverge across ranks and rank 0 would
    publish one arbitrary shard's running statistics.
    """
    count = 0
    for child in module.modules():
        if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            child.eval()
            if child.weight is not None:
                child.weight.requires_grad_(False)
            if child.bias is not None:
                child.bias.requires_grad_(False)
            count += 1
    return count


def configure_finetune_model(
    model: torch.nn.Module,
    *,
    train_scope: str,
    freeze_batchnorm: bool = True,
) -> int:
    """Select which model parameters are trainable during Tac-LeWM fine-tuning."""
    if train_scope not in {"encoder_only", "world_model"}:
        raise ValueError(
            "train_scope must be 'encoder_only' or 'world_model', "
            f"got {train_scope!r}"
        )
    if train_scope == "encoder_only":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        encoder = getattr(getattr(model, "jepa", None), "encoder", None)
        if encoder is None:
            raise ValueError("encoder_only fine-tuning requires model.jepa.encoder")
        encoder.requires_grad_(True)
    if freeze_batchnorm:
        return freeze_batchnorm_buffers(model)
    return 0


def set_finetune_train_mode(
    model: torch.nn.Module, *, freeze_batchnorm: bool
) -> None:
    """Enable training mode while keeping BatchNorm buffers frozen."""
    model.train()
    if freeze_batchnorm:
        freeze_batchnorm_buffers(model)


class FineTuneDataset(Dataset):
    def __init__(
        self, files: list[str], image_root: str, stats: dict, marker_cache_size: int = 256
    ):
        self.samples = EpisodeSampleIndex(
            (file, len(read_episode_rows(file))) for file in files
        )
        self.marker_loader = LoadImages(
            ["marker_left", "marker_right"], image_root, require_rgb=True
        )
        self.vision_loader = LoadImages(["images_1"], image_root)
        self.marker_cache = PreparedMarkerCache(marker_cache_size)
        self.mean = torch.tensor(stats["mean"], dtype=torch.float32)
        self.std = torch.tensor(stats["std"], dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file, t = self.samples[index]
        rows = read_episode_rows(file)
        indices = history_indices(t) + history_indices(t + 1)
        pairs = self.marker_cache.get_many(
            [(file, i) for i in indices],
            lambda key: self.marker_loader(dict(rows[key[1]]))["images"],
        )
        windows = [
            torch.stack(pairs[offset : offset + HISTORY], dim=1)
            for offset in (0, HISTORY)
        ]
        # The world model's vision preprocessing explicitly expects BGR bytes.
        vision = self.vision_loader(dict(rows[t]))["images"][0]
        vision = torch.from_numpy(np.asarray(vision)[..., ::-1].copy()).unsqueeze(0)
        action = torch.tensor(
            [rows[t]["action"], rows[t + 1]["action"]], dtype=torch.float32
        )
        return {
            "pixels": torch.stack(windows),
            "vision": vision,
            "action": (action - self.mean) / self.std,
            "source_id": torch.tensor(0),
        }


def objective(model, batch, regularizer, step: int, weight: float):
    output = model.encode(batch)
    embeddings = output["emb"]
    context, target = embeddings[:, :1], embeddings[:, 1:]
    prediction = model.predict(context, output["act_emb"][:, :1])
    prediction_target = target.detach() if model.pred_stopgrad else target
    mse = (prediction.float() - prediction_target.float()).square().mean()
    reg = regularizer(embeddings.float().transpose(0, 1))
    hinge = mse.new_zeros(())
    if model.training and model.pred_action_hinge:
        actions = output["act_emb"][:, :1]
        perm = torch.randperm(len(actions), device=actions.device)
        shuffled = actions[perm]
        changed = (actions - shuffled).abs().amax((1, 2)) > 0
        if changed.any():
            wrong = model.predict(context.detach(), shuffled.detach())
            hinge = pred_action_hinge(
                prediction[changed],
                wrong[changed],
                target[changed].detach(),
                margin=model.pred_action_hinge_margin,
            )
        scale = min(1.0, step / max(1, model.pred_action_hinge_warmup))
        hinge = hinge * scale * model.pred_action_hinge_weight
    loss = mse + weight * reg + hinge
    # Keep metrics on device until backward/optimizer work has been enqueued.
    return loss, dict(
        prediction_mse=mse.detach(), sigreg=reg.detach(),
        action_hinge=hinge.detach(), loss=loss.detach(),
    )


def training_metrics(metrics, gradient_norm, *, distributed: bool, weight: float):
    """Reduce detached losses and copy all log scalars to the host together."""
    names = ("prediction_mse", "sigreg", "action_hinge", "loss")
    values = torch.stack([metrics[name] for name in names]).to(torch.float64)
    if distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values.div_(dist.get_world_size())
    values = torch.cat((values, gradient_norm.detach().reshape(1).to(torch.float64)))
    result = dict(zip((*names, "gradient_norm"), values.cpu().tolist(), strict=True))
    if distributed:
        result["loss"] = (
            result["prediction_mse"] + weight * result["sigreg"] + result["action_hinge"]
        )
    return result


def _write_checkpoint(path, model, optimizer, step, config, resources, pixels):
    torch.save(model.state_dict(), path / "world_model.pt")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "step": step,
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state(pixels.device)
            if pixels.device.type == "cuda"
            else None,
        },
        path / "trainer.pt",
    )
    encoder_state = {
        "jepa.encoder." + key: value.detach().cpu()
        for key, value in model.encoder.state_dict().items()
    }
    torch.save(encoder_state, path / "encoder.pt")
    for name in (
        "world_model_config.json",
        "backbone_config.json",
        "vision_backbone_config.json",
    ):
        shutil.copyfile(resources / name, path / name)
    (path / "finetuning.json").write_text(json.dumps(config, indent=2))


def save_checkpoint(path, model, optimizer, step, config, resources, pixels):
    """Publish a checkpoint after all files have been written."""
    with staged_checkpoint(path) as staged:
        _write_checkpoint(
            staged, model, optimizer, step, config, resources, pixels
        )


def run(args):
    if (
        args.steps < 1
        or args.batch_size < 2
        or args.lr <= 0
        or args.validation_batches < 1
        or not np.isfinite(args.lr)
        or not np.isfinite(args.sigreg_weight)
        or args.sigreg_weight < 0
        or args.workers < 0
    ):
        raise ValueError(
            "Require steps>=1, batch_size>=2, lr>0 and validation_batches>=1"
        )
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        device = torch.device(args.device)
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    resources = args.resume or args.resources
    if args.resume:
        config = json.loads((args.resume / "finetuning.json").read_text())
        train, val = config["train_episodes"], config["validation_episodes"]
        stats = config["action_statistics"]
        weights = args.resume / "world_model.pt"
    else:
        train, val = split_episodes(args.jsonl_dir, args.validation_fraction, args.seed)
        stats = compute_action_statistics(train)
        weights = args.weights
        config = {
            "initial_weights": str(weights),
            "train_episodes": train,
            "validation_episodes": val,
            "action_statistics": stats,
            "learning_rate": args.lr,
            "sigreg_weight": args.sigreg_weight,
            "sequence": "two consecutive frames, each with 8x stride-5 marker history",
            "batch_size": args.batch_size,
            "seed": args.seed,
            "train_scope": args.train_scope,
            "freeze_batchnorm": args.freeze_batchnorm,
            "image_root": str(Path(args.image_dir).resolve()),
        }
    exists = torch.tensor(
        [int(args.output_dir.exists()) if rank == 0 else 0], device=device
    )
    if distributed:
        dist.broadcast(exists, src=0)
    if exists.item():
        if distributed:
            dist.destroy_process_group()
        raise FileExistsError(args.output_dir)
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    if distributed:
        dist.barrier(device_ids=[local_rank])
    model = load_world_model(weights, resources).to(device)
    train_scope = config["train_scope"]
    freeze_batchnorm = bool(config["freeze_batchnorm"])
    batchnorm_layers = configure_finetune_model(
        model,
        train_scope=train_scope,
        freeze_batchnorm=freeze_batchnorm,
    )
    # Preserve other source mappings exactly; only TacBench is used here.
    for name, projection in model.action_encoder.projs.items():
        if name != "tacbench":
            projection.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    start = 0
    if args.resume:
        state = torch.load(
            args.resume / "trainer.pt", map_location="cpu", weights_only=True
        )
        optimizer.load_state_dict(state["optimizer"])
        start = state["step"]
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        if device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state(state["cuda_rng"], device=device)
    marker_cache_size = getattr(args, "marker_cache_size", 256)
    train_data = FineTuneDataset(train, args.image_dir, stats, marker_cache_size)
    val_data = FineTuneDataset(val, args.image_dir, stats, marker_cache_size)
    if len(train_data) < args.batch_size * world_size:
        raise ValueError("Training split is smaller than batch_size")
    sampler = (
        DistributedSampler(
            train_data,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
        if distributed
        else None
    )
    loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=True,
        num_workers=args.workers,
        multiprocessing_context="spawn" if args.workers else None,
        pin_memory=device.type == "cuda",
    )
    validation = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=(
            DistributedSampler(
                val_data,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
            if distributed
            else None
        ),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_batches = max(
        1, (args.validation_batches + world_size - 1) // world_size
    )
    regularizer = SIGReg().to(device)
    epoch = 0
    if sampler is not None:
        sampler.set_epoch(epoch)
    iterator = iter(loader)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]

    def shutdown_loader_iterator(current_iterator):
        """Stop spawned workers before a distributed barrier or final validation."""
        shutdown = getattr(current_iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()

    logs = []
    set_finetune_train_mode(model, freeze_batchnorm=freeze_batchnorm)
    for step in range(start + 1, start + args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            shutdown_loader_iterator(iterator)
            del iterator
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            iterator = iter(loader)
            batch = next(iterator)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = objective(model, batch, regularizer, step, args.sigreg_weight)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite Tac-LeWM loss")
        loss.backward()
        if distributed:
            average_gradients(trainable)
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        metrics = training_metrics(
            metrics, norm, distributed=distributed, weight=args.sigreg_weight
        )
        metrics["step"] = step
        logs.append(metrics)
        if rank == 0:
            print(json.dumps(metrics), flush=True)
    shutdown_loader_iterator(iterator)
    del iterator
    del loader
    gc.collect()
    if distributed:
        dist.barrier(device_ids=[local_rank])

    model.eval()
    validation_error_sum = torch.zeros((), device=device, dtype=torch.float64)
    validation_count = torch.zeros((), device=device, dtype=torch.float64)
    sample_pixels = None
    with torch.no_grad():
        for batch_index, batch in enumerate(validation):
            if batch_index >= validation_batches:
                break
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            _, metrics = objective(model, batch, regularizer, step, args.sigreg_weight)
            count = len(batch["pixels"])
            validation_error_sum += metrics["prediction_mse"].to(torch.float64) * count
            validation_count += count
            if rank == 0:
                sample_pixels = batch["pixels"][:, 0]
    if distributed:
        validation_stats = torch.stack([validation_error_sum, validation_count])
        dist.all_reduce(validation_stats, op=dist.ReduceOp.SUM)
        validation_error_sum, validation_count = validation_stats
    error_sum, count = torch.stack([validation_error_sum, validation_count]).cpu().tolist()
    if count <= 0:
        raise RuntimeError("Validation produced no samples")
    if not np.isfinite(error_sum):
        raise RuntimeError("Validation produced a nonfinite result")
    if rank == 0:
        if sample_pixels is None:
            raise RuntimeError("Rank 0 did not receive a validation batch")
        validation_mse = error_sum / count
        config.update(
            training=logs,
            validation_mse=validation_mse,
            validation_batches=validation_batches * world_size
            if distributed
            else validation_batches,
            train_scope=train_scope,
            freeze_batchnorm=freeze_batchnorm,
            batchnorm_layers=batchnorm_layers,
            total_steps=step,
            distributed_world_size=world_size,
            resume_semantics="optimizer/RNG restored; shuffled data iterator restarts",
        )
        destination = args.output_dir / f"checkpoint-{step}"
        save_checkpoint(
            destination, model, optimizer, step, config, resources, sample_pixels
        )
        (args.output_dir / "report.json").write_text(json.dumps(config, indent=2))
        print(f"Encoder checkpoint: {destination / 'encoder.pt'}", flush=True)
    if distributed:
        dist.barrier(device_ids=[local_rank])
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl-dir", type=Path, required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resources", type=Path, default=RESOURCE)
    parser.add_argument("--weights", type=Path, default=RESOURCE / "weights_epoch_1.pt")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--sigreg-weight", type=float, default=0.09)
    parser.add_argument(
        "--marker-cache-size", type=int, default=256,
        help="Prepared frame pairs cached per data worker; 0 disables reuse.",
    )
    parser.add_argument(
        "--train-scope",
        choices=("encoder_only", "world_model"),
        default="encoder_only",
        help="Train only the tactile encoder by default; opt into full world-model tuning.",
    )
    parser.add_argument(
        "--allow-batchnorm-stats",
        action="store_true",
        help="Allow BatchNorm buffers/affine parameters to update (not recommended).",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--validation-batches", type=int, default=100)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.freeze_batchnorm = not args.allow_batchnorm_stats
    run(args)


if __name__ == "__main__":
    main()
