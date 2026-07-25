"""Model forward + masking smoke tests."""
import torch

from axle.models import build_model
from axle.models.backbones import masked_mean


def test_masked_mean_ignores_padding():
    x = torch.stack([torch.ones(4, 3), torch.zeros(4, 3)], dim=0)  # (2, 4, 3)
    mask = torch.tensor([[1.0, 1, 0, 0], [1, 1, 1, 1]])
    out = masked_mean(x, mask)
    assert torch.allclose(out[0], torch.ones(3))          # padded steps excluded
    assert torch.allclose(out[1], torch.zeros(3))


def test_all_backbones_forward_with_mask():
    b, t, c = 8, 24, 12
    x = torch.randn(b, t, c)
    mask = (torch.rand(b, t) > 0.4).float()
    mask[:, 0] = 1.0  # guarantee >=1 valid step
    for name in ["lstm", "tempcnn", "transformer"]:
        for var in (False, True):
            model = build_model(name, in_dim=c, predict_variance=var)
            out = model(x, mask)
            if var:
                assert set(out) == {"mu", "logvar"} and out["mu"].shape == (b,)
            else:
                assert out.shape == (b,)
