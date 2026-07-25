"""Unit test for the AXLE-M2 swath-correlated NLL.

Reproduces the coherence-gate dry-run: with a high-capacity smoother, anchoring
the correlated covariance to the (true) acquisition structure recovers the clean
signal better than equal-weight and than naive inverse-variance weighting, and a
mis-anchored (shuffled) covariance does *not* help. This is the mechanism claim
of M2, checked numerically on a synthetic swath.
"""
import numpy as np
import torch

from axle.losses.spatial import exponential_correlation, patch_covariance, correlated_nll


def test_exponential_correlation_shape_and_diagonal():
    coords = torch.tensor([[0.0, i] for i in range(5)])
    d = torch.tensor([0.0, 1.0])
    R = exponential_correlation(coords, d, torch.tensor(2.0))
    assert R.shape == (5, 5)
    assert torch.allclose(torch.diagonal(R), torch.ones(5), atol=1e-5)
    assert (R <= 1.0 + 1e-5).all() and (R >= 0).all()


def test_patch_covariance_is_spd():
    torch.manual_seed(0)
    k = 8
    coords = torch.stack([torch.zeros(k), torch.arange(k, dtype=torch.float32)], dim=1)
    cov = patch_covariance(torch.rand(k) + 0.1, torch.rand(k) + 0.1, coords, torch.tensor([0.0, 1.0]), torch.tensor(2.0))
    eig = torch.linalg.eigvalsh(cov)
    assert (eig > 0).all(), "covariance must be positive definite"


def test_correlated_nll_finite_and_differentiable():
    k = 6
    coords = torch.stack([torch.zeros(k), torch.arange(k, dtype=torch.float32)], dim=1)
    mu = torch.zeros(k, requires_grad=True)
    y = torch.randn(k)
    cov = patch_covariance(torch.ones(k) * 0.5, torch.ones(k) * 0.5, coords, torch.tensor([0.0, 1.0]), torch.tensor(1.5))
    nll = correlated_nll(y, mu, cov)
    assert torch.isfinite(nll)
    nll.backward()
    assert mu.grad is not None and torch.isfinite(mu.grad).all()


def test_anchoring_beats_naive_on_synthetic_swath():
    """The M2 mechanism claim, numerically (mirror of coh2.py)."""
    rng = np.random.default_rng(1)
    k = 40
    x = np.linspace(0, 1, k)
    f_true = 8 + 2 * np.sin(2 * np.pi * x)
    n = np.where((x > 0.4) & (x < 0.7), 1.0, 10.0)          # low-support stripe region
    s = np.full(k, 0.9)
    sigma2 = s**2 / np.maximum(n, 1)
    ell = 0.12
    D = np.abs(x[:, None] - x[None, :])
    R = np.exp(-D / ell)
    Sig = np.diag(np.sqrt(sigma2)) @ R @ np.diag(np.sqrt(sigma2))
    L2 = np.zeros((k - 2, k))                                # 2nd-difference smoother
    for i in range(k - 2):
        L2[i, i], L2[i, i + 1], L2[i, i + 2] = 1, -2, 1
    lam = 2.0

    def mc(prec, reps=800):
        errs = []
        for _ in range(reps):
            y = f_true + rng.multivariate_normal(np.zeros(k), Sig)
            mu = np.linalg.solve(prec + lam * L2.T @ L2, prec @ y)
            errs.append(np.sqrt(np.mean((mu - f_true) ** 2)))
        return np.mean(errs)

    equal = mc(np.eye(k))
    naive = mc(np.diag(1 / sigma2))
    ours = mc(np.linalg.inv(Sig + 1e-8 * np.eye(k)))
    assert ours < naive < equal, f"expected ours<naive<equal, got {ours:.3f},{naive:.3f},{equal:.3f}"
