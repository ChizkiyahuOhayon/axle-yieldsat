"""The Radon direction estimator must recover a planted swath angle, and must
*refuse* to invent one when the field has no stripes."""
import numpy as np
import pandas as pd

from axle.data.direction import estimate_direction, estimate_field_directions


def _striped_field(theta_deg: float, side: int = 40, swath: float = 5.0, seed: int = 0):
    """A square field whose support count is constant within parallel passes."""
    rng = np.random.default_rng(seed)
    r, c = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    r, c = r.ravel(), c.ravel()
    th = np.radians(theta_deg)
    across = -np.sin(th) * r + np.cos(th) * c
    pass_id = np.floor(across / swath).astype(int)
    level = rng.uniform(1, 20, size=pass_id.max() - pass_id.min() + 1)
    n_i = level[pass_id - pass_id.min()]
    return r, c, n_i


def _angle_error_deg(est: float, true: float) -> float:
    """Angular distance modulo 180 degrees (a stripe direction is unsigned)."""
    d = abs(est - true) % 180.0
    return min(d, 180.0 - d)


def test_recovers_planted_angle():
    for true_deg in (0.0, 30.0, 75.0, 120.0, 165.0):
        r, c, n = _striped_field(true_deg)
        est = estimate_direction(r, c, n)
        assert _angle_error_deg(est["theta_deg"], true_deg) <= 5.0, (true_deg, est)
        assert est["strength"] > 0.5, est
        # the returned vector is the unit vector of the reported angle
        assert np.isclose(np.hypot(est["dir_row"], est["dir_col"]), 1.0, atol=1e-5)


def test_no_stripes_gives_low_strength():
    rng = np.random.default_rng(3)
    r, c = np.meshgrid(np.arange(40), np.arange(40), indexing="ij")
    est = estimate_direction(r.ravel(), c.ravel(), rng.normal(10, 3, size=1600))
    assert est["strength"] < 0.2, est  # white noise explains almost nothing


def test_field_table_flags_and_zeroes_weak_fields():
    r, c, n = _striped_field(45.0)
    rng = np.random.default_rng(1)
    meta = pd.concat([
        pd.DataFrame({"field_shared_name": "striped", "row": r, "col": c, "n_i": n}),
        pd.DataFrame({"field_shared_name": "flat", "row": r, "col": c,
                      "n_i": rng.normal(10, 3, size=len(r))}),
    ], ignore_index=True)
    d = estimate_field_directions(meta, min_strength=0.3).set_index("field_shared_name")
    assert bool(d.loc["striped", "has_direction"])
    assert not bool(d.loc["flat", "has_direction"])
    assert (d.loc["flat", ["dir_row", "dir_col"]] == 0).all()  # isotropic fallback, not a guess
