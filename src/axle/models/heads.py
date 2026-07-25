"""Prediction heads.

A backbone produces a temporal embedding; the head maps it to the target.

* ``MeanHead``    -- point prediction (mu).  Output: (B,).
* ``MeanVarHead`` -- mean + log model-variance for heteroscedastic / AXLE losses.
                     Output: dict(mu=(B,), logvar=(B,)).
"""
from __future__ import annotations

import torch
from torch import nn


class MeanHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h).squeeze(-1)


class MeanVarHead(nn.Module):
    """Predicts mu and log sigma_model^2. ``softplus(logvar)`` is used downstream so
    the raw output is unconstrained and training is stable."""

    def __init__(self, dim: int):
        super().__init__()
        self.mu = nn.Linear(dim, 1)
        self.logvar = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor) -> dict:
        return {"mu": self.mu(h).squeeze(-1), "logvar": self.logvar(h).squeeze(-1)}
