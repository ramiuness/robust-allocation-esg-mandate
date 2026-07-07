"""Batch E gate — fit_predictor / OLS+Sigma internalization (#5 + #10 + OLS-dedup).

Veracity via an INDEPENDENT oracle (from-scratch numpy OLS), not against the deleted two-step:
if fit_predictor's B/Sigma match a plain numpy recomputation, the refactor is faithful. Plus the
fail-loud contract (forward/net_train before fit_predictor) and the base_rom pipeline end-to-end.
"""
import numpy as np
import pandas as pd
import pytest
import torch

import e2edro.DataLoad as dl
import e2edro.BaseModels as bm
import e2edro.e2edro as e2e


def _xy(n=40, n_x=12, n_y=4, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-06", periods=n, freq="W-MON")
    X = pd.DataFrame(rng.normal(size=(n, n_x)), index=idx, columns=[f"f{i}" for i in range(n_x)])
    Y = pd.DataFrame(rng.normal(size=(n, n_y)) * 0.02, index=idx, columns=[f"a{i}" for i in range(n_y)])
    return X, Y


def _numpy_ols(X, Y):
    """Independent OLS oracle: bias, weights from plain numpy lstsq on [ones | X]."""
    design = np.column_stack([np.ones(len(X)), X.values])
    theta = np.linalg.lstsq(design, Y.values, rcond=None)[0]     # [(1+n_x) x n_y]
    return theta[0], theta[1:].T                                  # bias (n_y,), weights (n_y,n_x)


def test_fit_predictor_B_matches_numpy_ols():
    X, Y = _xy()
    m = e2e.e2e_net(12, 4, 5, opt_layer="nominal", set_seed=0)
    m.fit_predictor(X, Y)
    bias, weight = _numpy_ols(X, Y)
    assert np.allclose(m.pred_layer.bias.detach().numpy(), bias, atol=1e-9)
    assert np.allclose(m.pred_layer.weight.detach().numpy(), weight, atol=1e-9)


def test_fit_predictor_sigma_matches_formula():
    X, Y = _xy()
    m = e2e.e2e_net(12, 4, 5, opt_layer="base_rom", set_seed=0)
    m.fit_predictor(X, Y)
    B = m.pred_layer.weight.detach().numpy()
    sigma_expected = B @ X.cov().values @ B.T          # Cov via pandas ddof=1
    assert np.allclose(m.sigma_mu_hat, sigma_expected, atol=1e-12)
    assert m.opt_layer is not None                     # built lazily by fit_predictor


def test_pred_then_opt_and_e2e_net_share_ols():
    # Both classes route OLS through e2e.ols_theta -> identical B for the same data.
    X, Y = _xy()
    a = e2e.e2e_net(12, 4, 5, opt_layer="nominal", set_seed=1)
    b = bm.pred_then_opt(12, 4, 5, opt_layer="nominal", set_seed=2)
    a.fit_predictor(X, Y)
    b.fit_predictor(X, Y)
    assert torch.allclose(a.pred_layer.weight, b.pred_layer.weight, atol=1e-12)


def test_forward_and_net_train_before_fit_raise():
    m = e2e.e2e_net(12, 4, 5, opt_layer="base_rom", set_seed=0)
    X = torch.randn(6, 12, dtype=torch.double)
    Y = torch.randn(5, 4, dtype=torch.double)
    with pytest.raises(RuntimeError):
        m(X, Y)                                        # forward before fit_predictor
    from torch.utils.data import DataLoader
    import e2edro.PortfolioClasses as pc
    Xdf, Ydf = _xy()
    ds = DataLoader(pc.SlidingWindow(Xdf, Ydf, 5, 2))
    with pytest.raises(RuntimeError):
        m.net_train(ds)                                # net_train before fit_predictor


def test_base_rom_roll_test_runs_and_fills():
    # Exercises fit_predictor inside net_roll_test + the guarded per-epoch _rebuild_opt_layer.
    rng = np.random.default_rng(3)
    n_tot, n_obs = 60, 5
    idx = pd.date_range("2020-01-06", periods=n_tot, freq="W-MON")
    X = pd.DataFrame(rng.normal(size=(n_tot, 12)), index=idx)
    Y = pd.DataFrame(rng.normal(size=(n_tot, 3)) * 0.02, index=idx)
    Xtt, Ytt = dl.TrainTest(X, n_obs, [0.6, 0.4]), dl.TrainTest(Y, n_obs, [0.6, 0.4])
    m = e2e.e2e_net(12, 3, n_obs, opt_layer="base_rom", perf_period=2, epochs=1, set_seed=0)
    m.net_roll_test(Xtt, Ytt, n_roll=2, epochs=1)
    w = m.portfolio.weights
    assert w.shape[0] == len(Ytt.test()) - n_obs
    assert int((np.abs(w).sum(axis=1) == 0).sum()) == 0
