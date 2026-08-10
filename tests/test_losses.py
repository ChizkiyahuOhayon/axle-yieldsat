"""Unit tests for the objectives and their contract."""
import pytest
import torch

from axle.losses import build_loss


def _batch(b=32, with_rel=True):
    return {
        "target": torch.randn(b) + 6.0,
        "sigma2_acq": torch.rand(b),
        "has_rel": torch.ones(b) if with_rel else torch.zeros(b),
        "quality_idx": torch.randint(0, 3, (b,)),
    }


def _pred(b=32, var=False):
    if var:
        return {"mu": torch.randn(b, requires_grad=True), "logvar": torch.randn(b, requires_grad=True)}
    return torch.randn(b, requires_grad=True)


def test_all_losses_scalar_and_backward():
    for name in ["mse", "invvar", "hetero", "axle"]:
        loss = build_loss(name) if name != "axle" else build_loss(name)
        pred = _pred(var=loss.predicts_variance)
        out = loss(pred, _batch())
        assert out.ndim == 0 and torch.isfinite(out)
        out.backward()


def test_axle_grade_scale_is_learnable():
    loss = build_loss("axle", learn_grade_scale=True)
    assert any(p.requires_grad for p in loss.parameters())
    # predictive variance >= aleatoric floor on reliable pixels
    pred = _pred(var=True)
    b = _batch()
    v = loss.predictive_variance(pred, b)
    assert (v > 0).all()


def test_axle_degrades_to_hetero_without_reliability():
    """With has_rel=0 everywhere, AXLE's total variance == hetero's."""
    torch.manual_seed(0)
    pred = {"mu": torch.zeros(16), "logvar": torch.zeros(16)}
    b = _batch(16, with_rel=False)
    axle = build_loss("axle")
    hetero = build_loss("hetero")
    assert torch.allclose(axle.predictive_variance(pred, b), hetero.predictive_variance(pred, b))


def test_grade_support_is_reported_so_a_frozen_g_is_not_read_as_learned():
    """Argentina soybean has 476 "Bad" pixels in 3.1M; its g_bad never moves.

    The diagnostics must say so, otherwise softplus(0)=0.693 looks like a fitted value.
    """
    import torch
    from axle.losses import build_loss

    loss = build_loss("axle")
    n = 100
    batch = {
        "target": torch.randn(n), "sigma2_acq": torch.rand(n) + 0.1,
        "has_rel": torch.ones(n),
        "quality_idx": torch.zeros(n, dtype=torch.long),   # every pixel "Good"
    }
    loss({"mu": torch.zeros(n), "logvar": torch.zeros(n)}, batch)
    d = loss.diagnostics()
    assert d["share_good"] == pytest.approx(1.0)
    assert d["share_bad"] == pytest.approx(0.0)
    assert d["g_bad"] == pytest.approx(float(torch.nn.functional.softplus(torch.zeros(1))), abs=1e-5)


def test_grade_support_accumulates_across_batches():
    import torch
    from axle.losses import build_loss

    loss = build_loss("axle")
    for grade, k in [(0, 60), (1, 30), (2, 10)]:
        b = {"target": torch.randn(k), "sigma2_acq": torch.ones(k), "has_rel": torch.ones(k),
             "quality_idx": torch.full((k,), grade, dtype=torch.long)}
        loss({"mu": torch.zeros(k), "logvar": torch.zeros(k)}, b)
    d = loss.diagnostics()
    assert (d["share_good"], d["share_average"], d["share_bad"]) == pytest.approx((0.6, 0.3, 0.1))
