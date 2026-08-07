r"""AXLE-M2 (experimental): swath-correlated Gaussian NLL over a field patch.

Harvester error is coherently striped along the machine's travel direction, so
the acquisition noise is *not* per-pixel independent. M2 models a field patch's
label noise with the covariance

.. math::  \Sigma = \mathrm{diag}(\sigma^2_{m}) + D^{1/2} R(\rho,\ell,d_f) D^{1/2},
           \qquad D=\mathrm{diag}(\sigma^2_{a}),

where :math:`R` is a Matern-1/2 (exponential) correlation along the harvester
direction :math:`d_f`, and trains with the correlated NLL
:math:`\tfrac12 (y-\mu)^\top\Sigma^{-1}(y-\mu) + \tfrac12\log|\Sigma|`.

The single-patch helpers below are the reference implementation (dense, readable,
unit-tested). :class:`SpatialAXLE` is the batched training objective that consumes
the field-patch loader of :mod:`axle.data.patches` and is selectable as
``loss=axle_spatial``.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .objectives import AXLE

_EPS = 1e-6


def _inv_softplus(x: float) -> float:
    """Value whose softplus is ``x`` -- for initialising positive parameters."""
    return math.log(math.expm1(x))


def exponential_correlation(coords: torch.Tensor, direction: torch.Tensor, ell: torch.Tensor) -> torch.Tensor:
    r"""Matern-1/2 correlation along ``direction`` with length ``ell``.

    Args:
        coords:    (k, 2) pixel (row, col) positions of a field patch, in metres/pixels.
        direction: (2,) unit vector of the harvester travel direction.
        ell:       correlation length (same units as coords).
    Returns:
        (k, k) correlation matrix with 1 on the diagonal.
    """
    d = direction / (direction.norm() + _EPS)
    proj = coords @ d                                   # (k,) along-track coordinate
    dist = (proj[:, None] - proj[None, :]).abs()        # (k, k) along-track distance
    return torch.exp(-dist / (ell + _EPS))


def patch_covariance(sigma2_model, sigma2_acq, coords, direction, ell) -> torch.Tensor:
    r"""Assemble :math:`\Sigma` for one field patch (dense; use for small patches / tests)."""
    r = exponential_correlation(coords, direction, ell)
    d = torch.sqrt(sigma2_acq + _EPS)
    sigma_acq = d[:, None] * r * d[None, :]             # D^{1/2} R D^{1/2}
    return torch.diag(sigma2_model + _EPS) + sigma_acq


def correlated_nll(y, mu, cov) -> torch.Tensor:
    r"""Multivariate-Gaussian NLL :math:`\tfrac12 r^\top\Sigma^{-1}r + \tfrac12\log|\Sigma|`.

    Uses a Cholesky solve; caller batches per field patch. For large patches,
    replace the Cholesky with the conjugate-gradient + stochastic-Lanczos-quadrature
    path described in ``docs/ROADMAP.md`` (the Toeplitz structure of ``R`` makes the
    matvec O(k log k)).
    """
    r = (y - mu).unsqueeze(-1)                          # (k, 1)
    L = torch.linalg.cholesky(cov)
    sol = torch.cholesky_solve(r, L)                    # Sigma^{-1} r
    quad = (r * sol).sum()
    logdet = 2.0 * torch.log(torch.diagonal(L)).sum()
    return 0.5 * (quad + logdet) / y.numel()


def pairwise_offsets(coords: torch.Tensor, direction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Split pixel separations into along-track and across-track components.

    The split is what makes the kernel *anisotropic*, and the anisotropy is the whole
    physical claim: error is coherent down a harvester pass and independent between
    passes. Using the along-track projection alone would say two pixels in different
    passes at the same along-track position are perfectly correlated -- the opposite of
    how a combine works.

    Args:
        coords:    (B, K, 2) pixel positions.
        direction: (B, 2) unit travel vectors; an all-zero row means "no stripe
                   resolved for this field", and the pair collapses to (Euclidean, 0),
                   i.e. an isotropic kernel with the along-track length scale.
    Returns:
        ``(along, across)``, each (B, K, K) and non-negative.
    """
    d = direction / (direction.norm(dim=-1, keepdim=True) + _EPS)      # (B, 2)
    perp = torch.stack([-d[:, 1], d[:, 0]], dim=-1)                    # (B, 2) across-track
    delta = coords[:, :, None, :] - coords[:, None, :, :]              # (B, K, K, 2)
    along = torch.einsum("bijc,bc->bij", delta, d).abs()
    across = torch.einsum("bijc,bc->bij", delta, perp).abs()

    has_dir = (direction.norm(dim=-1) > _EPS)[:, None, None]
    iso = torch.linalg.vector_norm(delta, dim=-1)
    return torch.where(has_dir, along, iso), torch.where(has_dir, across, torch.zeros_like(across))


class SpatialAXLE(AXLE):
    r"""AXLE-M2: swath-correlated NLL over field patches (a strict superset of M1).

    Per patch the label-noise covariance is

    .. math::  \Sigma = \mathrm{diag}(\sigma^2_{m}) + D^{1/2}\,R\,D^{1/2},\quad
               R = \rho\,e^{-\frac{\Delta_\parallel}{\ell_\parallel}-\frac{\Delta_\perp}{\ell_\perp}}
                   + (1-\rho) I,\quad D=\mathrm{diag}(\sigma^2_{a}),

    with the *same* anchored :math:`\sigma^2_{a}` as M1 (supplied by the harvester,
    scaled by the learnable grade factor :math:`g(q_f)`). Only the off-diagonal is new:
    at :math:`\rho=0` the matrix is diagonal and the objective is exactly
    :class:`~axle.losses.objectives.AXLE`, which makes M2-vs-M1 a one-parameter ablation
    rather than a different model.

    :math:`R` is a *product* of two Matern-1/2 kernels -- along the pass
    (:math:`\ell_\parallel`, long) and across it (:math:`\ell_\perp`, about one swath
    width) -- which is positive definite by the Schur product theorem. The anisotropy
    encodes the mechanism: one error draw is smeared down a pass, and the next pass is
    a fresh draw. :math:`\rho, \ell_\parallel, \ell_\perp` are learned in unconstrained
    space.

    Padded pixels are given an identity row/column and a zero residual, so they add
    nothing to either the quadratic form or the log-determinant.

    ``predictive_variance`` is inherited from M1: the *marginal* variance
    :math:`\sigma^2_{m}+\sigma^2_{a}` is what NLL/PICP should be scored against, and it
    keeps validation on the ordinary per-pixel path.
    """

    requires_patches = True

    def __init__(self, n_grades: int = 3, learn_grade_scale: bool = True,
                 ell_init: float = 8.0, ell_across_init: float = 3.0,
                 rho_init: float = 0.7, learn_kernel: bool = True, jitter: float = 1e-4):
        super().__init__(n_grades=n_grades, learn_grade_scale=learn_grade_scale)
        # store in unconstrained space: ell = softplus(raw), rho = sigmoid(raw)
        self.raw_ell = nn.Parameter(torch.tensor(_inv_softplus(ell_init)), requires_grad=learn_kernel)
        self.raw_ell_across = nn.Parameter(torch.tensor(_inv_softplus(ell_across_init)),
                                           requires_grad=learn_kernel)
        self.raw_rho = nn.Parameter(torch.tensor(math.log(rho_init / (1.0 - rho_init))),
                                    requires_grad=learn_kernel)
        self.jitter = float(jitter)

    @property
    def ell(self) -> torch.Tensor:
        """Correlation length along the pass, in pixels."""
        return F.softplus(self.raw_ell) + _EPS

    @property
    def ell_across(self) -> torch.Tensor:
        """Correlation length across passes -- about one swath width."""
        return F.softplus(self.raw_ell_across) + _EPS

    @property
    def rho(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_rho)

    @torch.no_grad()
    def diagnostics(self) -> dict:
        """M1's grade scales plus the learned swath geometry.

        ``ell_across`` is a falsifiable prediction: it should settle near one harvester
        swath width, and ``rho`` near 0 would say the correlation term earned nothing.
        """
        return {**super().diagnostics(), "rho": float(self.rho),
                "ell_along": float(self.ell), "ell_across": float(self.ell_across)}

    def covariance(self, pred, batch) -> torch.Tensor:
        """Assemble (B, K, K) :math:`\\Sigma`, with padded pixels neutralised to identity."""
        sigma2_model = F.softplus(pred["logvar"]) + _EPS                      # (B, K)
        sigma2_acq = batch["sigma2_acq"] * batch["has_rel"] * self.grade_scale(batch["quality_idx"])
        keep = batch["pix_mask"]                                              # (B, K)

        along, across = pairwise_offsets(batch["coords"], batch["direction"])
        eye = torch.eye(along.shape[-1], device=along.device, dtype=along.dtype)
        decay = torch.exp(-along / self.ell - across / self.ell_across)
        r = self.rho * decay + (1.0 - self.rho) * eye

        d = torch.sqrt(sigma2_acq + _EPS)
        cov = d[:, :, None] * r * d[:, None, :]
        cov = cov + torch.diag_embed(sigma2_model + self.jitter)

        # padded rows/cols -> identity: zero cross-terms, unit diagonal (log|.| += 0)
        pair = keep[:, :, None] * keep[:, None, :]
        return cov * pair + torch.diag_embed(1.0 - keep)

    def forward(self, pred, batch):
        if "coords" not in batch:  # guard: this objective is meaningless on a bag of pixels
            raise KeyError("axle_spatial needs field patches (coords/direction/pix_mask); "
                           "build the training set with axle.data.patches.YieldSATPatches")
        keep = batch["pix_mask"]
        resid = ((batch["target"] - pred["mu"]) * keep).unsqueeze(-1)          # (B, K, 1)
        cov = self.covariance(pred, batch)
        chol = torch.linalg.cholesky(cov)
        quad = (resid * torch.cholesky_solve(resid, chol)).sum()
        logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum()
        return 0.5 * (quad + logdet) / keep.sum().clamp_min(1.0)
