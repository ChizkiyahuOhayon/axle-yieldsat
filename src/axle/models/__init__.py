"""Model = backbone + head, assembled from config."""
from __future__ import annotations

import torch
from torch import nn

from . import backbones
from .heads import MeanHead, MeanVarHead

_BACKBONES = {"lstm": backbones.LSTM, "tempcnn": backbones.TempCNN, "transformer": backbones.Transformer}


class StaticEncoder(nn.Module):
    """Encode the per-pixel static context (soil / DEM / coordinates) for the head.

    These channels do not vary over the season, so feeding them through the temporal
    backbone 24 times would be pure waste; a small MLP whose output is concatenated to
    the pooled embedding gives the head the same information at a fraction of the cost.
    """

    def __init__(self, in_dim: int, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(inplace=True),
                                 nn.Linear(out_dim, out_dim))
        self.out_dim = out_dim

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


class YieldModel(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module, static_encoder: nn.Module | None = None):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.static_encoder = static_encoder

    def forward(self, x: torch.Tensor, mask: torch.Tensor, static: torch.Tensor | None = None):
        h = self.backbone(x, mask)
        if self.static_encoder is not None:
            if static is None:
                raise ValueError("model was built with static features but the batch has none")
            h = torch.cat([h, self.static_encoder(static)], dim=-1)
        return self.head(h)


def build_model(name: str, in_dim: int, predict_variance: bool, static_dim: int = 0,
                static_hidden: int = 32, **kw) -> YieldModel:
    """Assemble a model. ``predict_variance`` picks MeanVarHead (for hetero/AXLE) vs MeanHead.

    ``static_dim > 0`` adds a :class:`StaticEncoder` whose output is concatenated to the
    backbone embedding before the head.
    """
    if name not in _BACKBONES:
        raise ValueError(f"unknown backbone {name!r}; choose from {list(_BACKBONES)}")
    backbone = _BACKBONES[name](in_dim, **kw)
    static_encoder = StaticEncoder(static_dim, static_hidden) if static_dim else None
    dim = backbone.out_dim + (static_encoder.out_dim if static_encoder else 0)
    head = MeanVarHead(dim) if predict_variance else MeanHead(dim)
    return YieldModel(backbone, head, static_encoder)
