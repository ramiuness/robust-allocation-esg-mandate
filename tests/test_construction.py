"""Batch A gate — construction hardening (#8 double(), #7 max_weight, #3 dispatch).

These are the mechanical crash fixes; no modeling behavior changes. Kept solver-free except
one tiny forward pass that must run without an external model.double().
"""
import pytest
import torch

import e2edro.e2edro as e2e
import e2edro.BaseModels as bm

N_X, N_Y, N_OBS = 12, 4, 5
OPT_LAYERS = ["base_mod", "base_rom", "nominal", "tv", "hellinger"]


# --- #8: self.double() applied (was a no-op at top of __init__) ---------------------------
def test_e2e_net_params_are_double():
    m = e2e.e2e_net(N_X, N_Y, N_OBS, opt_layer="nominal", set_seed=0)
    assert m.pred_layer.weight.dtype == torch.double
    assert m.gamma.dtype == torch.double


def test_e2e_net_forwards_without_external_double():
    # #8: a freshly constructed model must forward without the caller doing model.double().
    m = e2e.e2e_net(N_X, N_Y, N_OBS, opt_layer="nominal", set_seed=0)
    X = torch.randn(N_OBS + 1, N_X, dtype=torch.double)
    Y = torch.randn(N_OBS, N_Y, dtype=torch.double)
    z, y_hat = m(X, Y)
    assert z.dtype == torch.double
    assert z.squeeze().shape == (N_Y,)


# --- #3: pred_then_opt dispatches every opt_layer (was NameError for nom/tv/hellinger) -----
@pytest.mark.parametrize("opt_layer", OPT_LAYERS)
def test_pred_then_opt_dispatches(opt_layer):
    po = bm.pred_then_opt(N_X, N_Y, N_OBS, opt_layer=opt_layer, set_seed=0)
    # base_rom builds its opt_layer lazily in fit_predictor (no placeholder); all others eager.
    if opt_layer == "base_rom":
        assert po.opt_layer is None
    else:
        assert po.opt_layer is not None


# --- #7: default max_weight is feasible; an infeasible cap raises up front -----------------
def test_e2e_net_default_max_weight_constructs():
    # default max_weight=1.0 (was None -> TypeError in the layer builder)
    e2e.e2e_net(N_X, N_Y, N_OBS, opt_layer="nominal", set_seed=0)


@pytest.mark.parametrize("ctor,kw", [
    (e2e.e2e_net, dict(opt_layer="nominal")),
    (bm.pred_then_opt, dict(opt_layer="base_mod")),
])
def test_infeasible_max_weight_raises(ctor, kw):
    # 0.1 * 4 assets < 1 -> no feasible budget; must raise at construction, not mid-solve.
    with pytest.raises(ValueError):
        ctor(N_X, N_Y, N_OBS, max_weight=0.1, set_seed=0, **kw)
