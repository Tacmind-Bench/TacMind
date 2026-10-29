"""EMA for the LeRobot training loop, with separate inference and resume weights."""
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path

import torch

from policy.adapters import temporary_attribute


@dataclass
class EMAConfig:
    # Diffusion Policy's power-law warmup schedule.
    decay: float = 0.9999
    inv_gamma: float = 1.0
    power: float = 0.75


class EMA:
    def __init__(self, model, config):
        self.config = config
        self.steps = 0
        self.shadow = {name: p.detach().clone() for name, p in model.named_parameters()}
        self._signature = None
        self._groups = {}

    @torch.no_grad()
    def update(self, model):
        self.steps += 1
        decay = min(self.config.decay, 1 - (1 + (self.steps - 1) / self.config.inv_gamma) ** -self.config.power)
        named = list(model.named_parameters())
        signature = tuple((name, id(p), p.device, p.dtype, p.requires_grad) for name, p in named)
        if signature != self._signature:
            groups = {}
            for name, parameter in named:
                value = self.shadow[name].to(device=parameter.device, dtype=parameter.dtype)
                self.shadow[name] = value
                key = (parameter.device, parameter.dtype, parameter.requires_grad)
                values, parameters = groups.setdefault(key, ([], []))
                values.append(value)
                parameters.append(parameter)
            self._groups, self._signature = groups, signature
        for (_, _, trainable), (values, parameters) in self._groups.items():
            if trainable:
                torch._foreach_lerp_(values, parameters, 1 - decay)
            else:
                torch._foreach_copy_(values, parameters)

    @contextmanager
    def applied(self, model):
        online = {name: p.detach().clone() for name, p in model.named_parameters()}
        try:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.copy_(self.shadow[name])
            yield
        finally:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.copy_(online[name])

    def save(self, checkpoint, model):
        from safetensors.torch import save_model

        folder = Path(checkpoint) / "training_state"
        folder.mkdir(parents=True, exist_ok=True)
        save_model(model, folder / "online_model.safetensors")
        (folder / "ema.json").write_text(json.dumps(dict(steps=self.steps)))

    def resume(self, checkpoint, model):
        from safetensors.torch import load_model

        folder = Path(checkpoint) / "training_state"
        metadata = json.loads((folder / "ema.json").read_text())
        self.steps = metadata["steps"]
        # make_policy loaded the inference (EMA) weights; keep those in shadow,
        # then restore the online weights that correspond to the saved optimizer.
        load_model(model, folder / "online_model.safetensors", strict=True)


@contextmanager
def training_ema(trainer, cfg):
    """Scope hooks to this training call; ACT and other policy types pass through."""
    original_make, original_update, original_save = trainer.make_policy, trainer.update_policy, trainer.save_checkpoint
    ema = None

    def make_policy(*args, **kwargs):
        nonlocal ema
        model = original_make(*args, **kwargs)
        if model.config.type == "diffusion":
            cfg.ema = getattr(cfg, "ema", None) or EMAConfig()
            ema = EMA(model, cfg.ema)
            if cfg.resume:
                ema.resume(cfg.checkpoint_path, model)
        return model

    def update(*args, **kwargs):
        result = original_update(*args, **kwargs)
        if ema is not None:
            # The official loop supplies policy positionally and accelerator by keyword.
            accelerator = kwargs["accelerator"]
            if not accelerator.optimizer_step_was_skipped:
                ema.update(accelerator.unwrap_model(args[1] if len(args) > 1 else kwargs["policy"]))
        return result

    def save(*args, **kwargs):
        if ema is None:
            return original_save(*args, **kwargs)
        model, checkpoint = kwargs["policy"], kwargs["checkpoint_dir"]
        ema.save(checkpoint, model)
        with ema.applied(model):
            return original_save(*args, **kwargs)

    with (temporary_attribute(trainer, "make_policy", make_policy),
          temporary_attribute(trainer, "update_policy", update),
          temporary_attribute(trainer, "save_checkpoint", save)):
        yield
