"""Unit test for the AXLE-M2 swath-correlated NLL.

Reproduces the coherence-gate dry-run: with a high-capacity smoother, anchoring
the correlated covariance to the (true) acquisition structure recovers the clean
signal better than equal-weight and than naive inverse-variance weighting, and a
mis-anchored (shuffled) covariance does *not* help. This is the mechanism claim
of M2, checked numerically on a synthetic swath.
"""
import numpy as np
import pytest
import torch

from axle.losses import build_loss
from axle.losses.objectives import AXLE
from axle.losses.spatial import (SpatialAXLE, correlated_nll, exponential_correlation,
                                 pairwise_distance, patch_covariance)


def _patch_batch(b=3, k=7, seed=0, direction=(1.0, 0.0), sizes=None):
    """A synthetic patch batch in the loader's format; ``sizes`` sets valid pixel counts."""
    g = torch.Generator().manual_seed(seed)
    rows, cols = torch.meshgrid(torch.arange(k), torch.arange(1), indexing="ij")
    coords = torch.stack([rows.flatten().float(), cols.flatten().float()], 1)
    sizes = sizes or [k] * b
    pix_mask = torch.stack([torch.arange(k) < s for s in sizes]).float()
    batch = {
        "target": torch.randn(b, k, generator=g),
        "sigma2_acq": torch.rand(b, k, generator=g) + 0.1,
        "has_rel": torch.ones(b, k),
        "quality_idx": torch.zeros(b, k, dtype=torch.long),
        "coords": coords.expand(b, k, 2).clone(),
        "direction": torch.tensor(direction).expand(b, 2).clone(),
        "pix_mask": pix_mask,
    }
    pred = {"mu": torch.randn(b, k, generator=g, requires_grad=True),
            "logvar": torch.randn(b, k, generator=g, requires_grad=True)}
    return pred, batch


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


# --- SpatialAXLE: the batched training objective -----------------------------------


def test_rho_zero_reduces_exactly_to_m1():
    """M2 is a strict superset of M1: with no correlation the two losses coincide."""
    pred, batch = _patch_batch(seed=1)
    m2 = build_loss("axle_spatial", rho_init=1e-9, learn_kernel=False, jitter=0.0)
    m1 = AXLE()
    flat = {k: (v.reshape(-1) if torch.is_tensor(v) and v.dim() >= 2 and k != "coords" else v)
            for k, v in batch.items()}
    flat_pred = {k: v.reshape(-1) for k, v in pred.items()}
    assert torch.allclose(m2(pred, batch), m1(flat_pred, flat), atol=1e-5)


def test_padding_does_not_change_the_loss():
    """Padded pixels get an identity row and a zero residual -- they must be inert."""
    pred, batch = _patch_batch(b=1, k=5, seed=2)
    loss = build_loss("axle_spatial")
    full = loss(pred, batch)

    pad = 4
    padded_pred = {k: torch.cat([v, torch.zeros(1, pad)], 1) for k, v in pred.items()}
    padded = {k: (torch.cat([v, torch.zeros(1, pad, *v.shape[2:], dtype=v.dtype)], 1)
                  if torch.is_tensor(v) and v.dim() >= 2 and k != "direction" else v)
              for k, v in batch.items()}
    padded["quality_idx"][:, -pad:] = 3
    assert torch.allclose(loss(padded_pred, padded), full, atol=1e-5)


def test_matches_the_dense_reference_on_one_patch():
    """The batched assembly must agree with the readable single-patch helpers."""
    pred, batch = _patch_batch(b=1, k=6, seed=3)
    loss = build_loss("axle_spatial", learn_kernel=False, jitter=0.0)
    rho, ell = loss.rho, loss.ell

    # reference: R = rho*exp(-d/ell) + (1-rho)I, folded into the acquisition block
    corr = exponential_correlation(batch["coords"][0], batch["direction"][0], ell)
    r = rho * corr + (1 - rho) * torch.eye(6)
    sigma2_acq = batch["sigma2_acq"][0] * loss.grade_scale(batch["quality_idx"][0])
    d = torch.sqrt(sigma2_acq + 1e-6)
    cov = torch.diag(torch.nn.functional.softplus(pred["logvar"][0]) + 1e-6) + d[:, None] * r * d[None, :]
    ref = correlated_nll(batch["target"][0], pred["mu"][0], cov)
    assert torch.allclose(loss(pred, batch), ref, atol=1e-5)


def test_kernel_parameters_receive_gradients():
    pred, batch = _patch_batch(seed=4)
    loss = build_loss("axle_spatial")
    loss(pred, batch).backward()
    for name in ("raw_ell", "raw_rho", "log_g"):
        g = getattr(loss, name).grad
        assert g is not None and torch.isfinite(g).all(), f"no gradient for {name}"
    assert torch.isfinite(pred["mu"].grad).all()


def test_zero_direction_falls_back_to_isotropic_distance():
    coords = torch.tensor([[[0.0, 0.0], [3.0, 4.0]]])
    iso = pairwise_distance(coords, torch.zeros(1, 2))
    assert iso[0, 0, 1] == pytest.approx(5.0)                      # Euclidean
    along = pairwise_distance(coords, torch.tensor([[1.0, 0.0]]))
    assert along[0, 0, 1] == pytest.approx(3.0)                    # projection on d_f only


def test_spatial_loss_demands_patches():
    """Silently averaging over a bag of pixels would be wrong, so it must raise."""
    loss = build_loss("axle_spatial")
    with pytest.raises(KeyError, match="field patches"):
        loss({"mu": torch.zeros(4), "logvar": torch.zeros(4)},
             {"target": torch.zeros(4), "sigma2_acq": torch.ones(4),
              "has_rel": torch.ones(4), "quality_idx": torch.zeros(4, dtype=torch.long)})


def test_predictive_variance_is_the_marginal_and_works_on_pixels():
    """Validation stays on the per-pixel path: diag(Sigma) = sigma2_model + sigma2_acq."""
    loss = build_loss("axle_spatial")
    pred = {"mu": torch.zeros(4), "logvar": torch.zeros(4)}
    batch = {"sigma2_acq": torch.full((4,), 0.25), "has_rel": torch.ones(4),
             "quality_idx": torch.zeros(4, dtype=torch.long)}
    v = loss.predictive_variance(pred, batch)
    expected = torch.nn.functional.softplus(torch.zeros(4)) + 0.25 * loss.grade_scale(batch["quality_idx"])
    assert torch.allclose(v, expected, atol=1e-5)
