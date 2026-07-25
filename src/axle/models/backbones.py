"""Temporal backbones for per-pixel yield regression.

All backbones consume ``(B, T, C)`` season-aligned pixel time series and emit a
``(B, dim)`` embedding; the head then maps it to the target. AXLE is
backbone-agnostic, so we keep a small, comparable set:

* ``LSTM``        -- the benchmark's recurrent baseline,
* ``TempCNN``     -- a 1D temporal CNN (Pelletier et al. style),
* ``Transformer`` -- a lightweight temporal encoder with learned CLS pooling.
"""
from __future__ import annotations

import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.rnn = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return out[:, -1, :]


class TempCNN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, layers: int = 3, kernel: int = 5, dropout: float = 0.2):
        super().__init__()
        chans = [in_dim] + [hidden] * layers
        blocks = []
        for a, b in zip(chans[:-1], chans[1:]):
            blocks += [nn.Conv1d(a, b, kernel, padding=kernel // 2), nn.BatchNorm1d(b),
                       nn.ReLU(inplace=True), nn.Dropout(dropout)]
        self.net = nn.Sequential(*blocks)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x.transpose(1, 2))       # (B, C, T)
        return h.mean(dim=-1)                  # global average pool over time


class Transformer(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, layers: int = 2, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
        self.pos = nn.Parameter(torch.zeros(1, 64, hidden))  # supports T up to 64 (>= 24)
        enc = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        h = self.proj(x) + self.pos[:, :t]
        h = torch.cat([self.cls.expand(b, -1, -1), h], dim=1)
        return self.encoder(h)[:, 0]          # CLS token
