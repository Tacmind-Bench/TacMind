"""OptimizerConfig: shared TacMind-0 runtime behavior."""

from dataclasses import dataclass
from typing import Literal

import torch

from tacmind0.optimizer import MuonAdamW, mark_muon_parameters
from tacmind0.optimizer.muon_adamw import is_default_muon_parameter


@dataclass
class OptimizerConfig:
    optim: Literal["adamw", "muon_adamw"] = "adamw"
    base_lr: float = 2.5e-5
    weight_decay: float = 1e-10
    warmup_steps: int = 1000
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_lr_scale: float = 1.0
    muon_moonlight_coefficient: float = 0.2

    def should_use_muon(
        self,
        name: str,
        param: torch.nn.Parameter,
    ) -> bool:
        return is_default_muon_parameter(name, param)

    def prepare_model(self, model: torch.nn.Module) -> None:
        if self.optim == "muon_adamw":
            mark_muon_parameters(model, predicate=self.should_use_muon)

    def build_muon_adamw(self, model: torch.nn.Module) -> MuonAdamW:
        return MuonAdamW(
            model,
            lr=self.base_lr,
            betas=(self.adam_beta1, self.adam_beta2),
            eps=self.adam_epsilon,
            weight_decay=self.weight_decay,
            muon_momentum=self.muon_momentum,
            muon_nesterov=self.muon_nesterov,
            muon_ns_steps=self.muon_ns_steps,
            muon_lr_scale=self.muon_lr_scale,
            muon_moonlight_coefficient=self.muon_moonlight_coefficient,
        )

    def _get_optimizer_grouped_parameters(self, model: torch.nn.Module) -> list:
        parameters = [
            {
                "params": [p for n, p in model.named_parameters() if p.requires_grad],
                "weight_decay": self.weight_decay,
            }
        ]
        return parameters
