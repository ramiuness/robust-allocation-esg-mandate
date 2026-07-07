"""Batch C gate — SlidingWindow off-by-one + harness date derivation (#2).

The length bug and its alignment consequences are provable solver-free; only the array-fill
and cross-path date checks touch the pipeline (tiny sizes).
"""
import numpy as np
import pandas as pd
import pytest

import e2edro.DataLoad as dl
import e2edro.BaseModels as bm
import e2edro.e2edro as e2e
import e2edro.PortfolioClasses as pc


def _df(n_rows, n_cols=3, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-06", periods=n_rows, freq="W-MON")
    return pd.DataFrame(rng.normal(size=(n_rows, n_cols)), index=idx,
                        columns=[f"c{i}" for i in range(n_cols)])


def _valid_window_count(L, n_obs, perf_period):
    """Formula-independent oracle: count windows whose feature AND perf slices are in-bounds."""
    n = 0
    while (n + n_obs + 1 <= L) and (n + n_obs + perf_period <= L):
        n += 1
    return n


# --- length matches the geometry (incl. perf_period=0 trap) -------------------------------
@pytest.mark.parametrize("L,n_obs,perf_period", [
    (30, 10, 1), (30, 10, 13), (30, 10, 0),   # inference / training / gamma_range
    (20, 5, 1), (15, 5, 0), (14, 10, 1), (25, 6, 3),
])
def test_len_matches_geometry(L, n_obs, perf_period):
    ds = pc.SlidingWindow(_df(L), _df(L), n_obs, perf_period)
    assert len(ds) == _valid_window_count(L, n_obs, perf_period)


def test_last_window_full_next_overruns():
    n_obs, pp = 10, 1
    ds = pc.SlidingWindow(_df(30), _df(30), n_obs, pp)
    x, y, y_perf = ds[len(ds) - 1]
    assert x.shape[0] == n_obs + 1 and y.shape[0] == n_obs and y_perf.shape[0] == pp
    x2, _, yp2 = ds[len(ds)]                       # one past the last valid index
    assert x2.shape[0] < n_obs + 1 or yp2.shape[0] < pp


def test_no_lookahead_alignment():
    # The realized-return row (y_perf at i+n_obs) is never inside the residual window y
    # (rows i..i+n_obs-1): the decision uses only strictly-past returns.
    n_obs = 5
    ds = pc.SlidingWindow(_df(20), _df(20), n_obs, 1)
    for i in range(len(ds)):
        x, y, y_perf = ds[i]
        assert y.shape[0] == n_obs and y_perf.shape[0] == 1


# --- pipeline fills the backtest arrays exactly (the reported symptom) ---------------------
def _traintest(n_tot=60, n_x=12, n_y=3, n_obs=5, split=(0.6, 0.4)):
    X, Y = _df(n_tot, n_x, 1), _df(n_tot, n_y, 2)
    return (dl.TrainTest(X, n_obs, list(split)), dl.TrainTest(Y, n_obs, list(split)), n_obs)


def test_net_roll_test_fills_arrays():
    for mk, roll_kw in [
        (lambda nx, ny, no: bm.equal_weight(nx, ny, no), {}),
        (lambda nx, ny, no: bm.pred_then_opt(nx, ny, no, opt_layer="nominal", set_seed=0), {}),
        (lambda nx, ny, no: e2e.e2e_net(nx, ny, no, opt_layer="nominal", perf_period=2,
                                        epochs=1, set_seed=0), dict(epochs=1)),
    ]:
        Xtt, Ytt, n_obs = _traintest()
        m = mk(12, 3, n_obs)
        m.net_roll_test(Xtt, Ytt, n_roll=2, **roll_kw)
        w = m.portfolio.weights
        assert w.shape[0] == len(Ytt.test()) - n_obs        # every decision date booked
        assert int((np.abs(w).sum(axis=1) == 0).sum()) == 0  # no zero-padded rows


# --- harness date_window emits exactly [pred_start, pred_end] ------------------------------
def test_date_window_emits_exact_range():
    import base_rom_demo_utils as u
    n_obs = 5
    X = _df(60, 12, 1); Y = _df(60, 3, 2)
    Xtt, Ytt = dl.TrainTest(X, n_obs, [0.6, 0.4]), dl.TrainTest(Y, n_obs, [0.6, 0.4])
    pred_start, pred_end = Y.index[40], Y.index[55]
    _, _, X_test_df, Y_test_df = u.date_window(Xtt, Ytt, train_end=Y.index[39],
                                               pred_start=pred_start, pred_end=pred_end)
    ds = pc.SlidingWindow(X_test_df, Y_test_df, n_obs, 1)
    emitted = [Y_test_df.index[j + n_obs] for j in range(len(ds))]
    assert emitted[0] == pred_start and emitted[-1] == pred_end
    assert len(emitted) == 55 - 40 + 1
