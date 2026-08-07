r"""AXLE objectives and their baselines.

Notation (per pixel :math:`i`):
    :math:`y_i`            observed yield (target),
    :math:`\mu_i`          predicted mean,
    :math:`\sigma^2_{m,i}` predicted model variance (softplus of the head's logvar),
    :math:`\sigma^2_{a,i}` acquisition variance = ``sigma2_acq`` :math:`\times g(q_i)`.

The Gaussian NLL used by ``hetero`` and ``axle`` is

.. math::  \tfrac12 \frac{(y_i-\mu_i)^2}{v_i} + \tfrac12 \log v_i,

with total variance :math:`v_i`. In ``hetero`` :math:`v_i=\sigma^2_{m,i}`; in AXLE
:math:`v_i=\sigma^2_{m,i}+\sigma^2_{a,i}` -- the model cannot lower the loss by
inflating its *own* variance on noisy pixels, because :math:`\sigma^2_{a,i}` is
supplied by the instrument, so it stops fitting swath artifacts.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

_EPS = 1e-6


def _as_mean(pred):
    """Accept either a bare mean tensor or a dict; return the mean."""
    return pred["mu"] if isinstance(pred, dict) else pred


class MSE(nn.Module):
    predicts_variance = False

    def forward(self, pred, batch):
        return F.mse_loss(_as_mean(pred), batch["target"])

    def predictive_variance(self, pred, batch):
        return None  # point predictor: no calibrated variance


class InverseVariance(nn.Module):
    """Fixed inverse-variance weighted L2 -- the *naive* anchored baseline.

    Down-weights unreliable pixels by ``1 / sigma2_acq`` but keeps an independent-
    noise assumption (no spatial correlation, no learned variance). Weights are
    normalised so the loss scale is comparable to MSE.
    """

    predicts_variance = False

    def forward(self, pred, batch):
        mu = _as_mean(pred)
        w = batch["has_rel"] / (batch["sigma2_acq"] + _EPS)
        w = torch.where(batch["has_rel"] > 0, w, torch.ones_like(w))  # missing -> weight 1
        w = w / (w.mean() + _EPS)
        return (w * (mu - batch["target"]) ** 2).mean()

    def predictive_variance(self, pred, batch):
        return None


class Heteroscedastic(nn.Module):
    """Learned heteroscedastic Gaussian NLL (variance fit from residuals, unanchored)."""

    predicts_variance = True

    def forward(self, pred, batch):
        v = F.softplus(pred["logvar"]) + _EPS
        return 0.5 * ((batch["target"] - pred["mu"]) ** 2 / v + torch.log(v)).mean()

    def predictive_variance(self, pred, batch):
        return F.softplus(pred["logvar"]) + _EPS


class AXLE(nn.Module):
    r"""AXLE-M1: acquisition-anchored heteroscedastic NLL.

    Total variance is the sum of the learned model variance and an aleatoric
    variance *anchored* to the harvester metadata,
    :math:`\sigma^2_{a,i} = \texttt{sigma2\_acq}_i \cdot g(q_i)`, where
    :math:`g` is a small, learnable per-quality-grade scale (Good/Average/Bad,
    plus a "missing" bucket fixed to 1). For pixels without a reliability signal
    (``has_rel == 0``) the aleatoric term vanishes and the objective degrades
    gracefully to :class:`Heteroscedastic`.
    """

    predicts_variance = True

    def __init__(self, n_grades: int = 3, learn_grade_scale: bool = True):
        super().__init__()
        # log-scale so g = softplus(param) stays positive; init g ~= 1.
        self.log_g = nn.Parameter(torch.zeros(n_grades), requires_grad=learn_grade_scale)

    def grade_scale(self, quality_idx: torch.Tensor) -> torch.Tensor:
        g = F.softplus(self.log_g) + _EPS                      # (n_grades,)
        g_full = torch.cat([g, g.new_ones(1)])                 # append "missing" bucket = 1
        return g_full[quality_idx.clamp(max=g_full.numel() - 1)]

    def _total_var(self, pred, batch):
        sigma2_acq = batch["sigma2_acq"] * batch["has_rel"] * self.grade_scale(batch["quality_idx"])
        return F.softplus(pred["logvar"]) + sigma2_acq + _EPS

    def forward(self, pred, batch):
        v = self._total_var(pred, batch)
        return 0.5 * ((batch["target"] - pred["mu"]) ** 2 / v + torch.log(v)).mean()

    def predictive_variance(self, pred, batch):
        return self._total_var(pred, batch)

    @torch.no_grad()
    def diagnostics(self) -> dict:
        """Learned parameters worth reporting next to the metrics.

        ``g`` says how much the harvester's own reliability grades scale the supplied
        variance: g(Bad) > g(Good) means the model *agrees* with the grading.
        """
        g = F.softplus(self.log_g) + _EPS
        return {f"g_{name}": float(v) for name, v in zip(("good", "average", "bad"), g)}
