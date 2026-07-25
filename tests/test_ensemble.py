"""Deep-ensemble aggregation: epistemic + aleatoric combination."""
import numpy as np
import pandas as pd

from axle.trainer import _aggregate_members


def _member(pred, var, seed_idx):
    return pd.DataFrame({
        "index": ["a", "b", "c"], "field_shared_name": ["f", "f", "g"],
        "n_i": [10.0, 3.0, 20.0], "target": [5.0, 6.0, 7.0],
        "prediction": pred, "variance": var,
    })


def test_point_loss_variance_is_epistemic_spread():
    dfs = [_member([5, 6, 7], [np.nan] * 3, 0), _member([5.2, 5.8, 7.4], [np.nan] * 3, 1)]
    agg = _aggregate_members(dfs)
    # mean prediction
    assert np.allclose(agg["prediction"].to_numpy(), [(5 + 5.2) / 2, (6 + 5.8) / 2, (7 + 7.4) / 2])
    # variance = spread of member means (aleatoric all-NaN -> epistemic only)
    expected = np.var([[5, 6, 7], [5.2, 5.8, 7.4]], axis=0)
    assert np.allclose(agg["variance"].to_numpy(), expected)


def test_variance_loss_adds_aleatoric_and_epistemic():
    dfs = [_member([5, 6, 7], [1.0, 1.0, 1.0], 0), _member([5, 6, 7], [3.0, 3.0, 3.0], 1)]
    agg = _aggregate_members(dfs)
    # identical means -> zero epistemic; aleatoric = mean(1,3)=2
    assert np.allclose(agg["variance"].to_numpy(), [2.0, 2.0, 2.0])


def test_alignment_is_by_index_not_row_order():
    d0 = _member([5, 6, 7], [1, 1, 1], 0)
    d1 = _member([5, 6, 7], [1, 1, 1], 1).iloc[::-1].reset_index(drop=True)  # shuffled rows
    agg = _aggregate_members([d0, d1])
    # aligned by index -> zero epistemic + aleatoric 1 = 1 (broken alignment would give [2,1,2])
    assert np.allclose(agg["variance"].to_numpy(), [1, 1, 1])
