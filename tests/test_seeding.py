"""Batch B gate — deterministic init (#4).

gamma/delta/epsilon are drawn from a per-model local torch.Generator, so the draw depends
only on the model's seed -- not on global RNG state or construction order -- and construction
no longer resets the global stream. Solver-free.
"""
import torch

import e2edro.e2edro as e2e


def _scalars(m):
    out = {"gamma": m.gamma.item()}
    if hasattr(m, "delta"):
        out["delta"] = m.delta.item()
    if hasattr(m, "epsilon"):
        out["epsilon"] = m.epsilon.item()
    return out


def test_same_seed_reproducible():
    a = e2e.e2e_net(12, 4, 5, opt_layer="tv", set_seed=7)
    b = e2e.e2e_net(12, 4, 5, opt_layer="tv", set_seed=7)
    assert _scalars(a) == _scalars(b)


def test_scalar_init_independent_of_global_state():
    # Build the SAME model under two DIFFERENT global RNG states; local generator makes the
    # scalar draws identical regardless (the core #4 property: order/state independence).
    torch.manual_seed(1)
    m1 = e2e.e2e_net(12, 4, 5, opt_layer="base_rom", set_seed=99)
    torch.manual_seed(2)
    torch.rand(5)  # perturb global further
    m2 = e2e.e2e_net(12, 4, 5, opt_layer="base_rom", set_seed=99)
    assert _scalars(m1) == _scalars(m2)


def test_construction_does_not_reset_global_stream():
    # Old code called torch.manual_seed(set_seed) inside __init__, which RESET the global
    # stream to a fixed point. New code must not: two builds with the same seed from different
    # prior global states leave the global stream in different places.
    torch.manual_seed(1)
    e2e.e2e_net(12, 4, 5, opt_layer="tv", set_seed=99)
    g1 = torch.rand(1).item()
    torch.manual_seed(2)
    e2e.e2e_net(12, 4, 5, opt_layer="tv", set_seed=99)
    g2 = torch.rand(1).item()
    assert g1 != g2


def test_order_independent():
    # A model's draw is independent of what was constructed before it.
    tv_first = e2e.e2e_net(12, 4, 5, opt_layer="tv", set_seed=3)
    hel_second = e2e.e2e_net(12, 4, 5, opt_layer="hellinger", set_seed=3)
    hel_first = e2e.e2e_net(12, 4, 5, opt_layer="hellinger", set_seed=3)
    tv_second = e2e.e2e_net(12, 4, 5, opt_layer="tv", set_seed=3)
    assert _scalars(tv_first) == _scalars(tv_second)
    assert _scalars(hel_first) == _scalars(hel_second)


def test_draws_within_bounds():
    assert 0.02 <= e2e.e2e_net(12, 4, 5, opt_layer="nominal", set_seed=5).gamma.item() <= 0.1
    assert 0.1 <= e2e.e2e_net(12, 4, 5, opt_layer="base_rom", set_seed=5).epsilon.item() <= 1.0
    n_obs = 20
    d = e2e.e2e_net(12, 4, n_obs, opt_layer="tv", set_seed=5).delta.item()
    assert (1 - 1 / n_obs) / 10 <= d <= (1 - 1 / n_obs) / 2
