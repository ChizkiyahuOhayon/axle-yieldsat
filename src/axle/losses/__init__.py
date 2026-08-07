"""Training objectives.

Every loss is an ``nn.Module`` with a uniform call signature::

    loss(pred, batch) -> scalar

where ``pred`` is the model output (a tensor of means, or a ``dict(mu, logvar)``
for variance-predicting heads) and ``batch`` carries ``target`` and the AXLE
reliability signals. ``.predicts_variance`` tells the trainer which head to build;
``.requires_patches`` tells it to feed field patches instead of a bag of pixels.

The objectives form the ablation of the AXLE contribution:

* ``mse``    -- equal-weight L2 (the benchmark's objective; every pixel equal),
* ``invvar`` -- fixed inverse-variance weighting (the *naive* anchored baseline:
                down-weights by 1/sigma2_acq but assumes independent noise),
* ``hetero`` -- learned heteroscedastic Gaussian NLL (variance fit from residuals,
                *unanchored* -- the ICLR'25-style baseline),
* ``axle``   -- AXLE-M1: heteroscedastic NLL whose aleatoric variance is *anchored*
                to the harvester's reported reliability (supplied, not fit),
* ``axle_spatial`` -- AXLE-M2: M1 plus the swath-correlated off-diagonal, trained on
                field patches; ``rho=0`` collapses it back to ``axle``.
"""
from __future__ import annotations

from .objectives import MSE, InverseVariance, Heteroscedastic, AXLE
from .spatial import SpatialAXLE

_LOSSES = {"mse": MSE, "invvar": InverseVariance, "hetero": Heteroscedastic,
           "axle": AXLE, "axle_spatial": SpatialAXLE}


def build_loss(name: str, **kw):
    if name not in _LOSSES:
        raise ValueError(f"unknown loss {name!r}; choose from {list(_LOSSES)}")
    return _LOSSES[name](**kw)
