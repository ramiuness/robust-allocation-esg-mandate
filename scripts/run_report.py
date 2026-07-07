"""RunReport: the harness-side recorder that the instrumented library emits to.

Implements e2edro.observability.Recorder. The library announces typed events (SolveEvent per
solve, EpochRecord per training epoch, on_window at each roll boundary, record_meta per model);
this class owns all the *policy* the library must not: cheap raw collection on the hot path, then
tidy pandas views + layer-aware diagnostics materialized on access.

Usage (single-window or net_roll_test, identical):
    report = RunReport()
    report.attach(models)              # sets model._recorder on each
    ... run (run_window / net_roll_test) ...
    report.solves(); report.learning(); report.warnings(); report.diagnostics(); report.summary()

Design: record_* are O(1) appends. Diagnostics are computed once per *failing* window inside
on_window (while the model's fitted state is still current — it moves on after the next window),
never on the hot solve path. Six accessors, kept separate, all joinable on (model, window).
"""
import os
import json
from collections import Counter
from dataclasses import asdict, replace

import numpy as np
import pandas as pd


class RunReport:
    def __init__(self):
        self._solves = []          # list[SolveEvent]
        self._epochs = []          # list[EpochRecord]
        self._windows = []         # list[dict] diagnostics for failing windows
        self._meta = {}            # model_name -> dict
        self._failed = set()       # {(model, window)} that saw a retry/fallback
        self._warn_counts = Counter()  # (model, phase, window, message) -> count

    # ---- attach ---------------------------------------------------------------
    def attach(self, models):
        """Point every model's recorder at this report. Returns self for chaining."""
        for m in (models.values() if isinstance(models, dict) else models):
            m._recorder = self
        return self

    # ---- Recorder protocol (hot path: cheap appends only) ---------------------
    def record_solve(self, event):
        # Tally warnings by (model, phase, window, message) so a warning repeated across many
        # optimal solves (e.g. a per-solve torch notice) becomes one counted row, not thousands.
        for w in event.warnings:
            self._warn_counts[(event.model, event.phase, event.window, w)] += 1
        if event.event == 'optimal' and event.warnings:
            event = replace(event, warnings=())      # counts kept above; free the strings
        self._solves.append(event)
        if event.event != 'optimal':
            self._failed.add((event.model, event.window))

    def record_epoch(self, record):
        self._epochs.append(record)

    def record_meta(self, model, **kw):
        self._meta.setdefault(getattr(model, '_name', None), {}).update(kw)

    def on_window(self, model, window, dates, X_train, Y_train):
        """Compute layer-aware diagnostics for this window iff it saw a retry/fallback.

        Done here (not lazily post-run) because model.B / sigma_mu_hat are current only until the
        next window refits. Cheap: failing windows are rare, and it is one small eigendecomposition.
        """
        name = getattr(model, '_name', None)
        if (name, window) not in self._failed:
            return
        diag = _window_diagnostics(model, X_train, Y_train)
        diag.update(model=name, window=window)
        self._windows.append(diag)

    # ---- tidy views (lazy; built on access) -----------------------------------
    def solves(self):
        """Aggregate per (model, phase, window): counts + solve_time distribution."""
        if not self._solves:
            return pd.DataFrame()
        df = pd.DataFrame(asdict(e) for e in self._solves)
        g = df.groupby(['model', 'phase', 'window'], dropna=False)
        out = g.agg(
            n_solves=('event', 'size'),
            n_optimal=('event', lambda s: (s == 'optimal').sum()),
            n_retry=('event', lambda s: (s == 'retry').sum()),
            n_fallback=('event', lambda s: (s == 'fallback').sum()),
            t_min=('solve_time', 'min'), t_mean=('solve_time', 'mean'),
            t_p50=('solve_time', 'median'),
            t_p95=('solve_time', lambda s: s.quantile(0.95) if s.notna().any() else np.nan),
            t_max=('solve_time', 'max'),
        )
        return out.reset_index()

    def failures(self):
        """One row per non-optimal solve, with exception + captured text (heavy cols kept as-is)."""
        rows = [asdict(e) for e in self._solves if e.event != 'optimal']
        return pd.DataFrame(rows)

    def warnings(self):
        """Deduped warnings with counts, per (model, phase, window, message). Nothing dropped;
        a warning repeated across N solves is one row with count=N (highest counts first)."""
        if not self._warn_counts:
            return pd.DataFrame()
        rows = [dict(model=m, phase=p, window=w, message=msg, count=c)
                for (m, p, w, msg), c in self._warn_counts.items()]
        return pd.DataFrame(rows).sort_values('count', ascending=False).reset_index(drop=True)

    def diagnostics(self):
        """One row per failing window: layer-aware conditioning + finiteness."""
        return pd.DataFrame(self._windows)

    def learning(self):
        """One row per (model, window, epoch): split losses, param path, grad/decay/theta norms."""
        return pd.DataFrame(asdict(r) for r in self._epochs)

    def meta(self):
        """One row per model: pred_loss_factor, weight_decay, ... (run metadata)."""
        return pd.DataFrame.from_dict(self._meta, orient='index')

    # ---- display + persistence ------------------------------------------------
    def summary(self):
        """Compact health table per model; a clean run is all zeros in the retry/fallback columns."""
        s = self.solves()
        if s.empty:
            return s
        agg = s.groupby('model').agg(
            n_solves=('n_solves', 'sum'), n_retry=('n_retry', 'sum'),
            n_fallback=('n_fallback', 'sum'), t_max=('t_max', 'max'),
        )
        d = self.diagnostics()
        agg['worst_cond'] = (d.groupby('model')['cond'].max() if not d.empty and 'cond' in d
                             else np.nan)
        w = self.warnings()
        agg['n_warnings'] = (w.groupby('model')['count'].sum() if not w.empty else 0)
        return agg.fillna({'n_warnings': 0}).reset_index()

    def save(self, run_dir, manifest=None):
        """Write the six views as CSV + a manifest.json + a pickle for exact reload."""
        os.makedirs(run_dir, exist_ok=True)
        for name, df in [('solves', self.solves()), ('failures', self.failures()),
                         ('warnings', self.warnings()), ('diagnostics', self.diagnostics()),
                         ('learning', self.learning()), ('meta', self.meta())]:
            df.to_csv(os.path.join(run_dir, f'{name}.csv'), index=(name == 'meta'))
        with open(os.path.join(run_dir, 'manifest.json'), 'w') as f:
            json.dump(manifest or {}, f, indent=2, default=str)
        pd.to_pickle(dict(solves=self._solves, epochs=self._epochs,
                          windows=self._windows, meta=self._meta), os.path.join(run_dir, 'raw.pkl'))
        return run_dir

    @classmethod
    def load(cls, run_dir):
        """Reconstruct a RunReport from a saved run_dir (exact, via the raw pickle)."""
        raw = pd.read_pickle(os.path.join(run_dir, 'raw.pkl'))
        r = cls()
        r._solves, r._epochs = raw['solves'], raw['epochs']
        r._windows, r._meta = raw['windows'], raw['meta']
        r._failed = {(e.model, e.window) for e in r._solves if e.event != 'optimal'}
        return r


def _window_diagnostics(model, X_train, Y_train):
    """Layer-aware conditioning + finiteness for one fitted window.

    base_rom -> spectrum of Sigma_mu_hat (the estimator covariance the SOCP uses);
    DRO layers -> spectrum of the residual scenario covariance Cov(ep). Finiteness on B/ep/Sigma
    catches the NaN/Inf root causes cheaply. Computed from the model's *current* fitted state.
    """
    B = model.pred_layer.weight.detach().cpu().numpy()
    b = model.pred_layer.bias.detach().cpu().numpy()
    Xv = X_train.values if hasattr(X_train, 'values') else np.asarray(X_train)
    Yv = Y_train.values if hasattr(Y_train, 'values') else np.asarray(Y_train)
    ep = Yv - (Xv @ B.T + b)
    diag = {'finite_B': bool(np.isfinite(B).all()), 'finite_ep': bool(np.isfinite(ep).all())}

    mtype = getattr(model, 'model_type', None)
    sigma = getattr(model, 'sigma_mu_hat', None)
    if mtype == 'base_rom' and sigma is not None:
        diag['kind'] = 'sigma_mu'
        diag['finite_sigma'] = bool(np.isfinite(np.asarray(sigma)).all())
        diag.update(_spectrum(np.asarray(sigma)))
    else:
        diag['kind'] = 'cov_ep'
        diag.update(_spectrum(np.cov(ep, rowvar=False)))
    return diag


def _spectrum(S, tol=1e-10):
    """rank / lam_max / lam_min / cond of a symmetric PSD matrix, mirroring base_rom's thin rule."""
    ev = np.linalg.eigvalsh(S)
    lam_max = float(ev.max())
    keep = ev[ev > tol * lam_max] if lam_max > 0 else ev
    lam_min = float(keep.min()) if len(keep) else float('nan')
    cond = float(lam_max / lam_min) if len(keep) and lam_min > 0 else float('inf')
    return {'rank': int(len(keep)), 'lam_max': lam_max, 'lam_min': lam_min, 'cond': cond}
