"""Temporal backbones for per-pixel yield regression.

All backbones consume ``(B, T, C)`` season-aligned pixel time series plus a
``(B, T)`` validity ``mask`` (1 = real observation, 0 = padding) and emit a
``(B, dim)`` embedding; the head then maps it to the target. AXLE is
backbone-agnostic, so we keep a small, comparable set:

* ``LSTM``        -- recurrent baseline (masked-mean over states),
* ``TempCNN``     -- 1D temporal CNN (Pelletier et al. style, masked-mean pool),
* ``Transformer`` -- temporal encoder with a padding mask and masked-mean pool.

Masking matters: YieldSAT's 24-slot frame is season-aligned, so real acquisitions
sit irregularly in the middle and both ends are padding. Pooling over valid steps
only (rather than reading the last step) is what lets these models fit the signal.
"""
from __future__ import annotations

import torch
from torch import nn


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``x`` (B, T, D) over the time steps where ``mask`` (B, T) is 1."""
    m = mask.unsqueeze(-1).to(x.dtype)                 # (B, T, 1)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


class LSTM(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.rnn = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return masked_mean(out, mask)                  # pool over valid steps, not the padded last step


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

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.net(x.transpose(1, 2)).transpose(1, 2)   # (B, T, hidden)
        return masked_mean(h, mask)


class Transformer(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, layers: int = 2, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.pos = nn.Parameter(torch.zeros(1, 64, hidden))  # supports T up to 64 (>= 24)
        enc = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        h = self.proj(x) + self.pos[:, :t]
        # padded steps are ignored by attention; keep any all-padding rows safe with clamp in pool
        h = self.encoder(h, src_key_padding_mask=(mask < 0.5))
        return masked_mean(h, mask)
