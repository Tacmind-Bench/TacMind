"""Train UniVTAC's native visuo-tactile DETRVAE with separate 9-state/8-action projections."""
from argparse import Namespace
import json
from pathlib import Path

import numpy as np
import torch

from policy.adapters import temporary_attribute
from policy.config import cameras
from policy.data import EpisodeDataset
from policy.checkpoint import atomic_checkpoint, bind_checkpoint, update_best


def build_model(cfg, *, initialize_backbones=True):
    from policy.act_tactile.detr.models import backbone
    from policy.act_tactile.detr.models.detr_vae import build
    options = dict(state_dim=8, chunk_size=cfg["chunk_size"], hidden_dim=512, dim_feedforward=3200,
                   position_embedding="sine", lr_vision_backbone=1e-5,
                   lr_tactile_backbone=1e-5, masks=False, dilation=False,
                   backbone="resnet18", nheads=8, enc_layers=4, dec_layers=7,
                   pre_norm=False, dropout=.1, camera_names=["agentview", "wristview"],
                   tactile_names=["gsmini_left", "gsmini_right"], tactile_type="feat",
                   tactile_masks=False, tactile_backbone="resnet18", tactile_ckpt=None)
    options.update(cfg["model"])
    # A complete policy checkpoint already contains the visual backbone. Avoid a
    # redundant ImageNet download when restoring it on an offline evaluation host.
    is_main_process = backbone.is_main_process if initialize_backbones else lambda: False
    with temporary_attribute(backbone, "is_main_process", is_main_process):
        model = build(Namespace(**options))
    # Keep all 9 measured joints without fabricating a ninth action target.
    hidden = options["hidden_dim"]
    model.input_proj_robot_state = torch.nn.Linear(9, hidden)
    model.encoder_joint_proj = torch.nn.Linear(9, hidden)
    return model.to(cfg["device"])


def statistics(dataset):
    result = {}
    for key in ("state", "action"):
        values = np.concatenate([arrays[key] for arrays in dataset.arrays.values()])
        result[key + "_mean"] = values.mean(axis=0)
        result[key + "_std"] = np.maximum(values.std(axis=0), .01)
    return result


def image_batch(images):
    from torchvision.transforms import functional as F
    if all(image.dtype == np.uint8 and image.shape == images[0].shape for image in images):
        # Stack bytes before conversion, avoiding a second full float-image copy.
        batch = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous()
        batch = batch.to(dtype=torch.get_default_dtype()).div_(255)
        return F.resize(batch, [256, 256], antialias=True)
    return torch.stack([F.resize(F.to_tensor(image), [256, 256], antialias=True) for image in images])


def inputs(sample, stats):
    from torchvision.transforms import functional as F
    state = torch.from_numpy((sample["state"] - stats["state_mean"]) / stats["state_std"]).float()
    views = cameras("act_tactile")
    vision = image_batch([sample[view] for view in views[:2]])
    tactile = image_batch([sample[view] for view in views[2:]])
    F.normalize(vision, [.485, .456, .406], [.229, .224, .225], inplace=True)
    return state, vision, tactile


class TrainingDataset:
    def __init__(self, dataset, stats):
        self.dataset, self.stats = dataset, stats

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        # Upstream's indexed dataset pads with zero targets before normalization.
        sample["actions"][sample["is_pad"]] = 0
        action = (sample["actions"] - self.stats["action_mean"]) / self.stats["action_std"]
        return (*inputs(sample, self.stats), torch.from_numpy(action).float(), torch.from_numpy(sample["is_pad"]))


def split_episodes(dataset, fraction, seed):
    """Use whole episodes for the upstream 80/20 split, never adjacent frames."""
    episodes = np.random.RandomState(seed).permutation(list(dataset.arrays)).tolist()
    count = int((1 - fraction) * len(episodes)) if fraction else len(episodes)
    split = dict(train=episodes[:count], validation=episodes[count:])
    indices = {}
    for name, ids in split.items():
        selected = set(ids)
        indices[name] = [i for row, start, stop in zip(
            dataset.manifest["episodes"], dataset.offsets[:-1], dataset.offsets[1:], strict=True)
            if row["index"] in selected for i in range(start, stop)]
    return indices, split


def batch_loss(model, batch, cfg):
    state, vision, tactile, actions, padding = [value.to(cfg["device"], non_blocking=True) for value in batch]
    predicted, _, (mu, logvar) = model(state, vision, tactile, None, actions, padding)
    l1 = ((predicted - actions).abs() * (~padding).unsqueeze(-1)).mean()
    kl = (-.5 * (1 + logvar - mu.square() - logvar.exp())).sum(1).mean()
    loss = l1 + cfg["model"].get("kl_weight", 10.) * kl
    return loss


def data_loader(dataset, cfg, *, shuffle):
    # Keep sampling and VAE randomness independent of worker startup/reuse.
    generator = torch.Generator().manual_seed(cfg["seed"])
    sampler = torch.utils.data.RandomSampler(dataset, generator=generator) if shuffle else None
    return torch.utils.data.DataLoader(
        dataset, batch_size=cfg["batch_size"], sampler=sampler,
        num_workers=cfg["num_workers"], persistent_workers=cfg["num_workers"] > 0,
        pin_memory=cfg["device"].startswith("cuda"),
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )


def train(cfg, manifest):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    dataset = EpisodeDataset(cfg["prepared_root"], cfg["chunk_size"], cameras("act_tactile"))
    indices, split = split_episodes(dataset, cfg.get("validation_fraction", .2), cfg["seed"])
    # Match upstream: fit normalization on all selected episodes.
    stats = statistics(dataset)
    transformed = TrainingDataset(dataset, stats)
    loader = data_loader(torch.utils.data.Subset(transformed, indices["train"]), cfg, shuffle=True)
    validation = data_loader(torch.utils.data.Subset(transformed, indices["validation"]), cfg, shuffle=False)
    model = build_model(cfg)
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if n.startswith("backbones.0") and p.requires_grad],
         "lr": cfg["model"].get("lr_vision_backbone", 1e-5)},
        {"params": [p for n, p in model.named_parameters() if n.startswith("backbones.1") and p.requires_grad],
         "lr": cfg["model"].get("lr_tactile_backbone", 1e-5)},
    ], lr=cfg["learning_rate"], weight_decay=1e-4)

    def save(step, validation_loss):
        from safetensors.torch import save_model
        checkpoint = Path(cfg["output_dir"]) / "checkpoints" / f"{step:06d}"
        with atomic_checkpoint(checkpoint) as staged:
            save_model(model, staged / "model.safetensors")
            np.savez(staged / "normalization.npz", **stats)
            (staged / "validation.json").write_text(json.dumps(
                dict(step=step, loss=validation_loss, episodes=split), indent=2))
            bind_checkpoint(staged, cfg, manifest)
        return checkpoint

    iterator = iter(loader)
    best_loss = float("inf")
    model.train()
    for step in range(1, cfg["steps"] + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        loss = batch_loss(model, batch, cfg)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step == 1:
            print(f"step={step} loss={loss.item():.6f}", flush=True)
        validation_loss = None
        if len(validation) and (step % len(loader) == 0 or step == cfg["steps"]):
            model.eval()
            # Compare checkpoints with the same VAE sampling noise, and leave the
            # training RNG untouched by validation or its DataLoader.
            devices = [torch.device(cfg["device"])] if cfg["device"].startswith("cuda") else []
            with torch.random.fork_rng(devices=devices), torch.inference_mode():
                torch.manual_seed(cfg["seed"])
                total = sum(batch_loss(model, batch, cfg).item() * len(batch[0]) for batch in validation)
            validation_loss = total / len(validation.dataset)
            print(f"step={step} validation_loss={validation_loss:.6f}", flush=True)
            model.train()
        improved = validation_loss is not None and validation_loss < best_loss
        if improved or step % cfg["save_freq"] == 0 or step == cfg["steps"]:
            checkpoint = save(step, validation_loss)
            if improved:
                update_best(checkpoint)
                best_loss = validation_loss


class Inference:
    def __init__(self, cfg, checkpoint):
        from safetensors.torch import load_model
        self.cfg = cfg
        # Full checkpoints include encoder weights; inference does not depend on the initialization path.
        cfg = {**cfg, "model": {**cfg["model"], "tactile_ckpt": None}}
        self.model = build_model(cfg, initialize_backbones=False)
        load_model(self.model, Path(checkpoint) / "model.safetensors", strict=True)
        self.model.eval()
        with np.load(Path(checkpoint) / "normalization.npz", allow_pickle=False) as data:
            self.stats = {key: data[key] for key in data.files}

    def infer(self, sample):
        state, vision, tactile = [value.unsqueeze(0).to(self.cfg["device"]) for value in inputs(sample, self.stats)]
        predicted, _, _ = self.model(state, vision, tactile, None)
        return predicted[0].cpu().numpy() * self.stats["action_std"] + self.stats["action_mean"]


def load(cfg, checkpoint):
    return Inference(cfg, checkpoint)
