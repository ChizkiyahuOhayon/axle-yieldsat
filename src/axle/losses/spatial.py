r"""AXLE-M2 (experimental): swath-correlated Gaussian NLL over a field patch.

Harvester error is coherently striped along the machine's travel direction, so
the acquisition noise is *not* per-pixel independent. M2 models a field patch's
label noise with the covariance

.. math::  \Sigma = \mathrm{diag}(\sigma^2_{m}) + D^{1/2} R(\rho,\ell,d_f) D^{1/2},
           \qquad D=\mathrm{diag}(\sigma^2_{a}),

where :math:`R` is a Matern-1/2 (exponential) correlation along the harvester
direction :math:`d_f`, and trains with the correlated NLL
:math:`\tfrac12 (y-\mu)^\top\Sigma^{-1}(y-\mu) + \tfrac12\log|\Sigma|`.

This module is standalone and unit-tested (``tests/test_spatial.py``); wiring it
into training needs a field-patch dataloader (see ``docs/ROADMAP.md``). It is not
yet part of the default ``train`` path.
"""
from __future__ import annotations

import torch

_EPS = 1e-6


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
