"""Offline tactile backbones shared by Tac-LeWM and TacMind-0."""

from torch import nn
from transformers import Dinov2WithRegistersConfig, Dinov2WithRegistersModel


class DinoV2RegisterBackbone(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.model = Dinov2WithRegistersModel(Dinov2WithRegistersConfig(**config))
