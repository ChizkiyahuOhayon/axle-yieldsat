"""Model = backbone + head, assembled from config."""
from __future__ import annotations

import torch
from torch import nn

from . import backbones
from .heads import MeanHead, MeanVarHead

_BACKBONES = {"lstm": backbones.LSTM, "tempcnn": backbones.TempCNN, "transformer": backbones.Transformer}


class YieldModel(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor):
        return self.head(self.backbone(x))


def build_model(name: str, in_dim: int, predict_variance: bool, **kw) -> YieldModel:
    """Assemble a model. ``predict_variance`` picks MeanVarHead (for hetero/AXLE) vs MeanHead."""
    if name not in _BACKBONES:
        raise ValueError(f"unknown backbone {name!r}; choose from {list(_BACKBONES)}")
    backbone = _BACKBONES[name](in_dim, **kw)
    head = MeanVarHead(backbone.out_dim) if predict_variance else MeanHead(backbone.out_dim)
    return YieldModel(backbone, head)
