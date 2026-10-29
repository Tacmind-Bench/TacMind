from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict
from sklearn.manifold import TSNE

from module import SIGReg
from utils import (
    SaveCkptCallback,
    build_dataset_from_cfg,
    get_artifacts_root,
    get_column_normalizer,
    get_dataset_tag_from_cfg,
    get_img_preprocessor,
    get_checkpoint_run_dir,
)

try:
    import seaborn as sns
except ImportError:  # optional plotting dependency
    sns = None


def _plot_and_save_tsne(tsne_embs, timesteps, actions, output_dir, step):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if sns is not None:
        sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 8))
    sc1 = plt.scatter(
        tsne_embs[:, 0],
        tsne_embs[:, 1],
        c=timesteps,
        cmap="viridis",
        s=25,
        alpha=0.8,
        edgecolors="none",
    )
    plt.colorbar(sc1, label="Batch Relative Time Step")
    plt.title(f"Latent t-SNE by Relative Timestep (step={step})")
    plt.tight_layout()
    plt.savefig(output_dir / f"tsne_by_timestep_step_{step:07d}.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 8))
    sc2 = plt.scatter(
        tsne_embs[:, 0],
        tsne_embs[:, 1],
        c=actions,
        cmap="magma",
        s=25,
        alpha=0.8,
        edgecolors="none",
    )
    plt.colorbar(sc2, label="Action Magnitude (L2 Norm)")
    plt.title(f"Latent t-SNE by Action Magnitude (step={step})")
    plt.tight_layout()
    plt.savefig(output_dir / f"tsne_by_action_step_{step:07d}.png", dpi=200)
    plt.close()


def _maybe_save_batch_tsne(module, emb, actions, stage, cfg):
    if stage != "fit":
        return
    if not cfg.get("tsne_every_n_steps", 100):
        return

    # step = int(module.global_step)
    step = module.trainer.global_step

    if step == 0 or (step % cfg.tsne_every_n_steps != 0):
        return

    if not module.trainer.is_global_zero:
        return

    # 取每个滑窗序列末帧，保持与 visualize_latent_tsne.py 一致的数据选择方式
    # numpy 不支持 bfloat16，先转换到 float32 再搬到 CPU
    last_frame_embs = emb[:, -1, :].detach().float().cpu().numpy()
    last_frame_actions = torch.norm(actions[:, -1, :], dim=-1).detach().float().cpu().numpy()

    # 避免 t-SNE perplexity 非法
    n_samples = last_frame_embs.shape[0]
    if n_samples < 4:
        return

    perplexity = int(cfg.get("tsne_perplexity", 40))
    perplexity = max(2, min(perplexity, n_samples - 1))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=cfg.seed,
        n_jobs=-1,
    )
    tsne_results = tsne.fit_transform(last_frame_embs)

    timesteps = np.arange(n_samples)
    output_dir = Path(module.run_dir) / "tsne_vis"
    _plot_and_save_tsne(tsne_results, timesteps, last_frame_actions, output_dir, step)


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight
    alpha = cfg.loss.get("patch_l2_weight", 0.0)

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    patch_emb = output.get("patch_emb", None)  # (B, T, P, D) if available
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss (CLS-level)
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()

    # Auxiliary patch-level L2 loss (optional, backward-compatible)
    if patch_emb is not None and alpha > 0:
        pred_patch_emb = pred_emb.unsqueeze(2).expand(-1, -1, patch_emb.size(2), -1)
        tgt_patch_emb = patch_emb[:, n_preds:]
        output["patch_l2_loss"] = (pred_patch_emb - tgt_patch_emb).pow(2).mean()
    else:
        output["patch_l2_loss"] = torch.zeros((), device=emb.device, dtype=emb.dtype)

    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + alpha * output["patch_l2_loss"] + lambd * output["sigreg_loss"]

    _maybe_save_batch_tsne(self, emb, batch["action"], stage, cfg)

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset = build_dataset_from_cfg(cfg.data.dataset, cache_dir=cfg.get("cache_dir", None))
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    dataset_tag = get_dataset_tag_from_cfg(cfg.data.dataset)
    run_name = f"{cfg.output_model_name}_{dataset_tag}"

    run_id = cfg.get("subdir") or ""
    run_dir = get_checkpoint_run_dir(run_name, run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    world_model.run_dir = str(run_dir)

    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=run_name, cfg=cfg.model, epoch_interval=1,
        cache_dir=str(get_artifacts_root()),
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()
