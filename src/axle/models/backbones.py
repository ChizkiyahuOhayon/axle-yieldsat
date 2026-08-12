"""Temporal backbones for per-pixel yield regression.

All backbones consume ``(B, T, C)`` season-aligned pixel time series plus a
``(B, T)`` validity ``mask`` (1 = real observation, 0 = padding) and emit a
``(B, dim)`` embedding; the head then maps it to the target. AXLE is
backbone-agnostic, so we keep a small, comparable set:

* ``LSTM``        -- recurrent baseline (masked-mean over states),
* ``TempCNN``     -- 1D temporal CNN (Pelletier et al. style, masked-mean pool),
* ``Transformer`` -- temporal encoder with a padding mask and masked-mean pool,
* ``Spatial3D``   -- 3D-CNN over a field tile (space *and* time), the family the
  YieldSAT benchmark reports as its strongest. It consumes tiles rather than loose
  pixels (``consumes_patches = True``) and still emits one embedding per pixel, so
  every head, loss and metric downstream is unchanged.

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


class Spatial3D(nn.Module):
    """3D-CNN over a field tile: (B, K=tile^2, T, C) -> (B, K, hidden).

    The benchmark's key finding is that spatial context matters -- its 3D-CNN models
    beat every temporal-only baseline. This is the smallest architecture that gives a
    pixel both its own season *and* its neighbours', while keeping AXLE's per-pixel
    objective intact: convolutions run over (time, row, col), then time is pooled away
    with the same masked mean the temporal backbones use, leaving one vector per pixel.

    Fields are irregular, so a tile is only ~58-69% occupied. Empty cells arrive
    zero-filled and are excluded from the loss by ``pix_mask``; ``valid`` additionally
    prevents them from leaking into their neighbours' features.
    """

    consumes_patches = True  # trainer feeds (B, K, T, C) tiles, not flattened pixels

    def __init__(self, in_dim: int, tile: int = 16, hidden: int = 64, layers: int = 3,
                 kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        self.tile = int(tile)
        pad = kernel // 2
        chans = [in_dim] + [hidden] * layers
        blocks = []
        for a, b in zip(chans[:-1], chans[1:]):
            blocks += [nn.Conv3d(a, b, kernel, padding=pad), nn.BatchNorm3d(b),
                       nn.ReLU(inplace=True), nn.Dropout3d(dropout)]
        self.net = nn.Sequential(*blocks)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                pix_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, k, t, c = x.shape
        h = w = self.tile
        if k != h * w:
            raise ValueError(f"Spatial3D expects K=tile^2={h * w} pixels per item, got {k}")

        valid = torch.ones(b, k, device=x.device, dtype=x.dtype) if pix_mask is None else pix_mask
        grid = (x * valid[:, :, None, None]).reshape(b, h, w, t, c).permute(0, 4, 3, 1, 2)
        feat = self.net(grid)                                   # (B, hidden, T, H, W)

        feat = feat.permute(0, 3, 4, 2, 1).reshape(b * k, t, -1)  # (B*K, T, hidden)
        pooled = masked_mean(feat, mask.reshape(b * k, t))
        return pooled.reshape(b, k, -1)
