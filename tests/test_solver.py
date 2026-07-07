"""Batch F gate — owned solve strategy, ZERO monkeypatch (#6).

Solver-free: the strict->retry->fallback logic is tested via a stub opt_layer (no cone solve),
and the arg wiring via plain dict/attribute assertions.
"""
import pytest
import torch

import e2edro.e2edro as e2e
import e2edro.BaseModels as bm

OPT_LAYERS = ["base_mod", "base_rom", "nominal", "tv", "hellinger"]


# --- fail-loud default: '*_inacc' collapsed onto strict ------------------------------------
@pytest.mark.parametrize("model_type", ["base_mod", "base_rom", "nom", "dro"])
def test_default_inacc_equals_strict(model_type):
    a = e2e.default_solver_args(model_type)
    for k in ("abstol", "reltol", "feastol"):
        assert a[f"{k}_inacc"] == a[k]
    assert a["verbose"] is False


# --- PO and SPO use identical solver_args per opt_layer (divergence closed) -----------------
@pytest.mark.parametrize("opt_layer", OPT_LAYERS)
def test_po_and_spo_solver_args_match(opt_layer):
    spo = e2e.e2e_net(12, 4, 5, opt_layer=opt_layer, set_seed=0)
    po = bm.pred_then_opt(12, 4, 5, opt_layer=opt_layer, set_seed=0)
    assert spo.solver_args == po.solver_args


def test_pred_then_opt_not_bare_ecos():
    # Previously pred_then_opt ran bare ECOS (max_iters=100); now it carries the tuned default.
    po = bm.pred_then_opt(12, 4, 5, opt_layer="base_mod", set_seed=0)
    assert po.solver_args["max_iters"] >= 10000


def test_user_solver_args_override():
    custom = {"solve_method": "ECOS", "max_iters": 123, "abstol": 1e-5}
    m = e2e.e2e_net(12, 4, 5, opt_layer="nominal", set_seed=0, solver_args=custom)
    assert m.solver_args is custom


# --- robust_solve strict -> retry -> fallback (stub opt_layer, no solver) -------------------
class _Stub:
    def __init__(self, fail_strict, fail_retry, fallback=True):
        self.n_y = 3
        self.solver_args = {"tag": "strict"}
        self.solve_retry_args = {"tag": "retry"}
        self.solve_fallback = fallback
        self._solve_log = []
        self._solve_phase = "infer"
        self.pred_layer = type("P", (), {"weight": torch.zeros(1)})()
        self._fail_strict, self._fail_retry = fail_strict, fail_retry

    def opt_layer(self, *params, solver_args=None):
        if solver_args is self.solver_args and self._fail_strict:
            raise RuntimeError("strict failed")
        if solver_args is self.solve_retry_args and self._fail_retry:
            raise RuntimeError("retry failed")
        return (torch.ones(self.n_y, 1),)


def test_robust_solve_strict_ok():
    s = _Stub(fail_strict=False, fail_retry=False)
    (z,) = e2e.robust_solve(s)
    assert z.shape == (3, 1) and s._solve_log == []          # no event on clean strict solve


def test_robust_solve_retry_logged():
    s = _Stub(fail_strict=True, fail_retry=False)
    e2e.robust_solve(s)
    assert s._solve_log == [("infer", "retry")]


def test_robust_solve_fallback_equal_weight():
    s = _Stub(fail_strict=True, fail_retry=True)
    (z,) = e2e.robust_solve(s)
    assert torch.allclose(z, torch.full((3, 1), 1.0 / 3, dtype=torch.double))
    assert s._solve_log == [("infer", "fallback")]


def test_robust_solve_raises_without_fallback():
    s = _Stub(fail_strict=True, fail_retry=True, fallback=False)
    with pytest.raises(RuntimeError):
        e2e.robust_solve(s)
