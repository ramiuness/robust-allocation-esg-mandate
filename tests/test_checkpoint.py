"""Batch D gate — init-state checkpoint (#1).

The pristine snapshot is now an in-memory, per-object deepcopy of state_dict (no shared on-disk
file), so same-model_type models (tv/hellinger) cannot clobber each other's init. Solver-free.
"""
import torch

import e2edro.e2edro as e2e


def _mk(opt_layer, n_obs=20, seed=7):
    return e2e.e2e_net(12, 4, n_obs, opt_layer=opt_layer, set_seed=seed)


def test_init_state_present_and_disk_removed():
    m = _mk("nominal")
    assert isinstance(m._init_state, dict) and "gamma" in m._init_state
    assert not hasattr(m, "init_state_path")   # the shared on-disk checkpoint is gone


def test_tv_hellinger_distinct_and_no_cross_mutation():
    tv = _mk("tv")
    tv_delta_before = tv._init_state["delta"].clone()
    hel = _mk("hellinger")            # same config/seed, both model_type='dro'
    # distinct init (tv/hellinger draw delta from different bounds)
    assert not torch.equal(tv._init_state["delta"], hel._init_state["delta"])
    # constructing hellinger did not touch tv's snapshot (old code shared one disk file)
    assert torch.equal(tv._init_state["delta"], tv_delta_before)


def test_reset_restores_scalar():
    m = _mk("tv")
    init_delta = m._init_state["delta"].clone()
    with torch.no_grad():
        m.delta.data.add_(0.123)     # perturb as training would
    assert not torch.equal(m.delta.data, init_delta)
    m.load_state_dict(m._init_state)
    assert torch.equal(m.delta.data, init_delta)


def test_cv_pickle_key_distinct():
    # The CV results pickle is keyed on opt_layer_name (+ pred_model + seed), so tv/hellinger
    # no longer collide on '<model_type>_results.pkl'.
    assert _mk("tv").opt_layer_name != _mk("hellinger").opt_layer_name
