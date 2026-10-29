"""N0-VTLA native PyTorch trainer with a portable TacMind episode loader."""
from contextlib import ExitStack
from dataclasses import dataclass, replace
import importlib
from pathlib import Path

import numpy as np

from policy.adapters import temporary_attribute, tensor_batches
from policy.config import cameras
from policy.data import EpisodeDataset
from policy.checkpoint import bind_checkpoint


PACKAGE = "n0vtla"


@dataclass(frozen=True)
class Inputs:
    def __call__(self, sample):
        image = dict(base_0_rgb=sample["agentview"], left_wrist_0_rgb=sample["wristview"],
                     right_wrist_0_rgb=np.zeros_like(sample["agentview"]))
        mask = dict(base_0_rgb=np.True_, left_wrist_0_rgb=np.True_, right_wrist_0_rgb=np.False_)
        for source, target in (("gsmini_left", "tactile_a"), ("gsmini_right", "tactile_b")):
            image[target] = sample[source]
            image[target + ".baseline"] = sample[source + ".baseline"]
            mask[target] = mask[target + ".baseline"] = np.True_
        result = dict(state=sample["state"], image=image, image_mask=mask, prompt=sample["prompt"])
        if "actions" in sample:
            result["actions"] = sample["actions"]
        return result


@dataclass(frozen=True)
class Outputs:
    def __call__(self, data):
        return {"actions": np.asarray(data["actions"])[..., :8]}


@dataclass(frozen=True)
class DataFactory:
    norm_stats: dict

    def create(self, assets_dirs, model_config):
        config = importlib.import_module(PACKAGE + ".training.config")
        transforms = importlib.import_module(PACKAGE + ".transforms")
        # The native joint recipes learn arm deltas, leaving the gripper absolute.
        # TacMind stores absolute targets, so convert before computing/using stats
        # and invert after native unnormalization at inference.
        mask = transforms.make_bool_mask(7, -1)
        data_transforms = transforms.Group(inputs=[Inputs()], outputs=[Outputs()]).push(
            inputs=[transforms.DeltaActions(mask)], outputs=[transforms.AbsoluteActions(mask)])
        return config.DataConfig(
            repo_id="tacmind", asset_id="tacmind", norm_stats=self.norm_stats,
            data_transforms=data_transforms,
            model_transforms=config.ModelTransformFactory()(model_config),
            use_quantile_norm=model_config.pi05,
        )


def make_config(cfg, stats):
    name = PACKAGE
    config = importlib.import_module(name + ".training.config")
    base = config.get_config("sim_single_arm_tactile")
    model = replace(base.model, **{"action_horizon": cfg["chunk_size"], "pytorch_compile_mode": None, **cfg["model"]})
    return replace(base, name="tacmind", exp_name="run", model=model,
                   data=DataFactory(stats), seed=cfg["seed"],
                   batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
                   num_train_steps=cfg["steps"], save_interval=cfg["save_freq"],
                   checkpoint_base_dir=str(Path(cfg["output_dir"]) / "checkpoints"),
                   assets_base_dir=str(Path(cfg["output_dir"]) / "assets"),
                   pytorch_weight_path=cfg.get("pretrained_path"),
                   lr_schedule=replace(base.lr_schedule, peak_lr=cfg["learning_rate"]),
                   wandb_enabled=False, overwrite=False, resume=False)


def compute_statistics(dataset):
    normalize = importlib.import_module(PACKAGE + ".shared.normalize")
    transforms = importlib.import_module(PACKAGE + ".transforms")
    delta = transforms.DeltaActions(transforms.make_bool_mask(7, -1))
    accumulators = {key: normalize.RunningStats() for key in ("state", "actions")}
    # Visit every training frame; preserve the same tail-clamped action distribution as the loader.
    for arrays in dataset.arrays.values():
        accumulators["state"].update(arrays["state"])
        for offset in range(dataset.horizon):
            indices = np.minimum(np.arange(len(arrays["action"])) + offset, len(arrays["action"]) - 1)
            transformed = delta(dict(state=arrays["state"], actions=arrays["action"][indices, None].copy()))
            accumulators["actions"].update(transformed["actions"][:, 0])
    return {key: value.get_statistics() for key, value in accumulators.items()}


def train(cfg, manifest):
    name = PACKAGE
    dataset = EpisodeDataset(cfg["prepared_root"], cfg["chunk_size"], cameras(cfg["policy_type"]),
                             tactile_baselines=True)
    stats = compute_statistics(dataset)
    config = make_config(cfg, stats)
    loader = importlib.import_module(name + ".training.data_loader")
    with ExitStack() as overrides:
        overrides.enter_context(temporary_attribute(loader, "create_torch_dataset", lambda *args, **kwargs: dataset))
        native = importlib.import_module(name + ".models_pytorch.pi0_pytorch")
        model = importlib.import_module(name + ".models_pytorch.n0vtla_policy").N0VTLAPolicy
        overrides.enter_context(temporary_attribute(native, "PI0Pytorch", model))
        trainer = importlib.import_module("scripts.train_pytorch")
        original_save = trainer.save_checkpoint

        def save(model, optimizer, global_step, train_config, is_main, data_config):
            # Upstream's final-step predicate differs by version; force a native save at the requested final step.
            effective = replace(train_config, save_interval=1) if global_step == cfg["steps"] else train_config
            original_save(model, optimizer, global_step, effective, is_main, data_config)
            path = train_config.checkpoint_dir / str(global_step)
            if is_main and (path / "model.safetensors").exists():
                bind_checkpoint(path, cfg, manifest)

        overrides.enter_context(temporary_attribute(trainer, "save_checkpoint", save))
        overrides.enter_context(tensor_batches(loader))
        trainer.train_loop(config)


class Inference:
    def __init__(self, cfg, checkpoint):
        name = PACKAGE
        normalize = importlib.import_module(name + ".shared.normalize")
        stats = normalize.load(Path(checkpoint) / "assets" / "tacmind")
        native = importlib.import_module(name + ".policies.policy_config")
        # N0's permissive loader is useful for base-model fine-tuning, but a
        # benchmark checkpoint must include its trained tactile branch as well.
        self.policy = native.create_trained_policy(make_config(cfg, stats), checkpoint,
                                                   norm_stats=stats, pytorch_device=cfg["device"])

    def infer(self, sample):
        return self.policy.infer(sample)["actions"]

    def reset(self):
        # PyTorch sampling uses the worker's reseeded torch RNG; no cross-episode action queue here.
        if hasattr(self.policy, "reset"):
            self.policy.reset()


def load(cfg, checkpoint):
    return Inference(cfg, checkpoint)
