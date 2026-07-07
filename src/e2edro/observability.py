"""Observability contract for the solve / learning recorder.

Defines ONLY the data the library emits (SolveEvent, EpochRecord) and the Recorder Protocol the
harness implements. The library depends on nothing here beyond dataclasses/typing; all
aggregation, diagnostics, DataFrame construction and persistence live in the harness recorder
(see RunReport). This keeps the split clean: library = event emission, harness = policy.

Emit sites (the only four): robust_solve (per solve), net_train (per epoch),
net_roll_test / infer window boundary (per window), calibrate_models (per-model meta). Every emit
is guarded by `model._recorder is not None`, so a model with no recorder is byte-identical to the
pre-instrumentation library.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class SolveEvent:
    """One cone-solve attempt.

    Heavy fields (traceback / solver_text / verbose_log) are populated only on non-optimal
    events; `warnings` may be present on any solve (e.g. diffcp's 'Solved/Inaccurate.').
    """
    model: Optional[str]
    phase: Optional[str]
    window: Optional[int]
    event: str                       # 'optimal' | 'retry' | 'fallback'
    date: Any = None
    solve_time: Optional[float] = None
    canon_time: Optional[float] = None
    shapes: Any = None
    warnings: Sequence[str] = ()
    exc_type: Optional[str] = None
    exc_msg: Optional[str] = None
    traceback: Optional[str] = None
    solver_text: Optional[str] = None
    verbose_log: Optional[str] = None


@dataclass(frozen=True)
class EpochRecord:
    """One training epoch's learning telemetry (Adam steps once per epoch in net_train)."""
    model: Optional[str]
    window: Optional[int]
    epoch: int
    loss_total: Optional[float] = None
    loss_task: Optional[float] = None
    loss_pred: Optional[float] = None
    loss_val: Optional[float] = None
    gamma: Optional[float] = None
    delta: Optional[float] = None
    epsilon: Optional[float] = None
    grad_norm_pred: Optional[float] = None
    grad_norm_robust: Optional[float] = None
    decay_norm: Optional[float] = None
    theta_l2: Optional[float] = None
    theta_dist_l2: Optional[float] = None


@runtime_checkable
class Recorder(Protocol):
    """Sink the library emits to; implemented by the harness RunReport.

    The library calls these only when `model._recorder is not None`. `on_window` receives the
    window's standardized train slice so the recorder can compute lazy diagnostics (Sigma_mu_hat
    spectrum, Cov(ep) conditioning, finiteness) for windows that saw a retry/fallback.
    """
    def record_solve(self, event: SolveEvent) -> None: ...
    def record_epoch(self, record: EpochRecord) -> None: ...
    def on_window(self, model: Any, window: int, dates: Any,
                  X_train: Any, Y_train: Any) -> None: ...
    def record_meta(self, model: Any, **kw: Any) -> None: ...
