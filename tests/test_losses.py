"""Unit tests for the objectives and their contract."""
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
