"""
Shared experiment + plotting harness for the ilo-portfolio-allocation notebooks
(base_rom_demo, esg_experiment_*, esg_alloc_*, e2e_ro_vs_dro_*, esg_diagnostic).

Contents
- Data loading (load_data), feature-relation diagnostics (feature_diagnostics,
  feature_variants, select_best_subset) and metrics (portfolio_metrics,
  build_metrics_table, running_dd).
- Plotting: wealth curves and drawdown/summary bars are Plotly (plot_wealth —
  aliased plot_all_wealth for the all-models overview — plot_drawdown,
  plot_summary_bars); the single-panel static diagnostics stay matplotlib
  (plot_epsilon_trajectory, plot_weight_heatmap).
- Experiment helpers (build_models, calibrate_models, fit_window, infer_window,
  date_window, run_window, run_feature_sweep, trained_params, model_report) that
  abstract model-zoo instantiation and the decoupled fit/inference flow out of the
  notebook cells.

Import patterns in use: `from base_rom_demo_utils import *` (pulls in np, pd,
torch, dl + every helper, keeping setup cells short) and explicit
`from base_rom_demo_utils import (...)`.
"""
import os as _os

# CUDA-deterministic reductions must be configured before torch initializes CUDA.
# The DRO cone programs (esp. TV) are ill-conditioned enough that the ~1e-7 GPU
# reduction noise amplifies to full-cap weight swings; together with the per-layer
# solver args + reject-inaccurate wrapper below and use_deterministic_algorithms()
# in set_seeds(), this makes same-seed/same-data runs bit-reproducible
# (see spo-critical-review.md Part II).
_os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import torch
from torch.utils.data import DataLoader

import e2edro.BaseModels as bm
import e2edro.PortfolioClasses as pc
import e2edro.DataLoad as dl
from e2edro.e2edro import e2e_net
from e2edro.progress import track

from run_report import RunReport

_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))


def set_seeds(seed=42):
    """Seed numpy and torch, and enable deterministic algorithms. Returns the seed.

    Deterministic mode removes the last nondeterminism source in the train-once/
    allocate workflow — CUDA reductions — so B/y_hat/ep are bit-identical across
    runs and the (deterministic) ECOS/diffcp solves return identical weights for
    the same seed and data. warn_only=True keeps it from erroring on any op that
    lacks a deterministic CUDA kernel (none of the ops in this pipeline do, but it
    is the safe default). See spo-critical-review.md Part II.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return seed


# ---------------------------------------------------------------------------
# Solver configuration  (see spo-critical-review.md Part II). The robust solve strategy
# (strict -> relaxed retry -> equal-weight fallback, with the '*_inacc' band collapsed so an
# inaccurate solve raises instead of being silently accepted) now lives IN the library as
# e2e.robust_solve, configured via constructor args. The harness only *supplies* the per-opt_layer
# settings below to build_models -- there is NO CvxpyLayer.forward monkeypatch.
# ---------------------------------------------------------------------------

# The '*_inacc' thresholds are set EQUAL to the strict tolerances on purpose: this
# collapses ECOS's "inaccurate" fallback band so a solve that cannot reach the strict
# tolerance RAISES (diffcp only warns on flag-10 otherwise, and cvxpylayers would
# silently accept the noisy iterate) — which the robust wrapper then catches and
# retries at _RETRY_ARGS. Without this, reject-inaccurate never engages.
def _ecos(tol, max_iters):
    return {'solve_method': 'ECOS', 'max_iters': max_iters, 'verbose': False,
            'abstol': tol, 'reltol': tol, 'feastol': tol,
            'abstol_inacc': tol, 'reltol_inacc': tol, 'feastol_inacc': tol}

SOLVER_ARGS = {
    'base_mod':  _ecos(1e-10, 10000),
    'base_rom':  _ecos(1e-9,  10000),
    'nominal':   _ecos(1e-8,  20000),
    'tv':        _ecos(1e-8,  20000),
    'hellinger': _ecos(1e-8,  20000),
}
# Relaxed retry for a window that cannot reach its strict tolerance.
_RETRY_ARGS = {'solve_method': 'ECOS', 'max_iters': 50000, 'verbose': False,
               'abstol': 1e-6, 'reltol': 1e-6, 'feastol': 1e-6}

# Event tags recorded per solve and read back by solve_report().
_EVENT_RETRY = 'retry'        # solve reached only the relaxed tolerance
_EVENT_FALLBACK = 'fallback'  # solve failed even relaxed -> equal-weight z
_PHASES = ('calibrate', 'train', 'infer')

# Solve events are recorded by e2e.robust_solve into model._solve_log as (phase, event); the
# phase is set on each model via `model._solve_phase = phase` before each solve batch (see
# calibrate_models / fit_window / infer_window). No module-level context or monkeypatch is needed.


def solve_report(models):
    """Per-model table of solver retries and fallbacks, split by phase.

    A clean run is all zeros. Nonzero 'infer' entries are the allocation windows that
    could not be solved to strict tolerance — retried at the relaxed 1e-6, or (for a
    fallback) replaced by equal weight. See spo-critical-review.md Part II.
    """
    labels = {_EVENT_RETRY: 'retry', _EVENT_FALLBACK: 'fallback'}
    rows = {}
    for name, model in models.items():
        log = getattr(model, '_solve_log', [])
        rows[name] = {f'{label}_{phase}': log.count((phase, event))
                      for event, label in labels.items() for phase in _PHASES}
    return pd.DataFrame(rows).T


def load_data(start='2020-01-02', end='2025-12-31', n_y=20, n_obs=104,
              split=(0.6, 0.4), freq='weekly', data_dir=None, features=None):
    """Load features/returns from the repo's local CSVs (data_dir auto-resolved).

    start, end : ISO date bounds (clamped to data availability inside the loader).
    n_y        : number of assets (alphabetical) or None for all.
    n_obs      : rolling-window size.
    split      : train/test ratio.
    freq       : 'weekly' or 'daily'.
    data_dir   : override; default <repo>/data.
    features   : feature selection passed to fetch_data_from_disk. None/'all' ⇒
                 all 12 features; else a str or list mixing group names
                 ('ff5+mom', 'macro', 'esg') and/or individual column names.
    returns    : (X, Y) TrainTest objects from fetch_data_from_disk. Features are
                 RAW (bp-scale yields, decimal returns); standardization happens
                 later at each train-window boundary (date_window / calibrate_models).
    """
    data_dir = data_dir or _os.path.join(_REPO_ROOT, 'data')
    return dl.fetch_data_from_disk(start, end, split=list(split), freq=freq,
                                   n_obs=n_obs, n_y=n_y, features=features,
                                   data_dir=data_dir)


# Names re-exported by `from base_rom_demo_utils import *` (keeps the notebook
# namespace clean while still giving it np/pd/torch/dl + every helper).
# portfolio_metrics / running_dd are intentionally omitted: they are internal
# helpers behind build_metrics_table / plot_drawdown and no notebook imports them.
__all__ = [
    'np', 'pd', 'torch', 'dl', 'COLORS',
    'set_seeds', 'load_data',
    'build_metrics_table',
    'plot_wealth', 'plot_epsilon_trajectory', 'plot_weight_heatmap',
    'plot_all_wealth', 'plot_drawdown', 'plot_summary_bars', 'plot_epsilon_sweep',
    'build_models', 'calibrate_models', 'fit_window', 'infer_window',
    'date_window', 'run_window', 'run_feature_sweep', 'run_epsilon_sweep',
    'feature_variants', 'feature_diagnostics', 'select_best_subset', 'cv_score_subset',
    'trained_params', 'model_report', 'roll_param_trajectory',
    'SOLVER_ARGS', 'solve_report', 'RunReport',
]

# ---------------------------------------------------------------------------
# Color / style registry
# ---------------------------------------------------------------------------
COLORS = {
    'EW'           : ('grey',        '--'),
    'PO-M'         : ('navy',        '--'),
    'PO-ROM'       : ('steelblue',   '--'),
    'SPO-M'        : ('dodgerblue',  '-'),
    'SPO-ROM'      : ('crimson',     '-'),
    'SPO-Nominal'  : ('darkorange',  '-'),
    'SPO-TV'       : ('purple',      '-'),
    'SPO-Hellinger': ('forestgreen', '-'),
}

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def portfolio_metrics(name, p):
    """Return annualised performance metrics for a backtest portfolio object."""
    ann = np.sqrt(p.periods_per_year)          # frequency-aware annualization (was hardcoded 52)
    ann_vol  = p.vol * ann
    rets_arr = p.rets['rets'].values
    down     = rets_arr[rets_arr < 0]
    ann_down = np.std(down) * ann if len(down) > 0 else np.nan
    sharpe   = p.annualized_return / ann_vol if ann_vol > 0 else np.nan
    sortino  = p.annualized_return / ann_down if (ann_down and ann_down > 0) else np.nan
    return {
        'Model'          : name,
        'Ann. Return (%)': round(p.annualized_return * 100, 2),
        'Ann. Vol (%)'   : round(ann_vol * 100, 2),
        'Sharpe'         : round(sharpe, 3),
        'Sortino'        : round(sortino, 3),
        'Max DD (%)'     : round(p.max_drawdown * 100, 2),
        'Ann. Turnover'  : round(p.turnover * p.periods_per_year, 2),
        'Eff. Holdings'  : round(p.effective_holdings, 2),
    }


def build_metrics_table(names, portfolios):
    """Build a DataFrame of annualised performance metrics for a list of models."""
    return pd.DataFrame(
        [portfolio_metrics(n, p) for n, p in zip(names, portfolios)]
    ).set_index('Model')


def running_dd(portfolio):
    """Compute running drawdown series from a portfolio object."""
    tri  = portfolio.rets['tri']
    peak = tri.expanding().max()
    return (tri - peak) / peak


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------
def _wealth_anchor(index):
    """Prepend a baseline wealth=1.0 anchor one period before the first test date.

    Mirrors the library's wealth_plot so curves start at 1.0 rather than 1+r0.
    Returns `index` with one extra leading timestamp (or integer) at the front.
    """
    anchor = (index[0] - pd.Timedelta(days=7)
              if isinstance(index, pd.DatetimeIndex) else index[0] - 1)
    return index.insert(0, anchor)


# plot_wealth (cumulative-wealth curves) is Plotly and lives in the interactive
# section below. plot_epsilon_trajectory and plot_weight_heatmap stay matplotlib:
# they are single-panel static diagnostics (a bar chart and an imshow grid), not the
# shared time-series overlay that benefits from Plotly's hover/legend interactivity.
def plot_epsilon_trajectory(spo_rom, n_roll, figsize=(8, 4)):
    """Bar chart of learned epsilon per roll window for an SPO-ROM model."""
    windows = list(range(1, n_roll + 1))
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(windows, spo_rom.epsilon_trained,
           color='crimson', alpha=0.75, label='Learned ε (end of window)')
    ax.axhline(
        spo_rom.epsilon_init, color='black', linestyle='--', linewidth=1.5,
        label=f'Initial ε = {spo_rom.epsilon_init:.3f}'
    )
    ax.set_xlabel('Roll window')
    ax.set_ylabel('ε')
    ax.set_title('SPO-ROM: Learned ε across roll windows')
    ax.set_xticks(windows)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_weight_heatmap(portfolios, asset_labels=None, names=None, ncols=2, figsize=None):
    """Grid of allocation-weight heatmaps — one panel per portfolio.

    portfolios : dict {label: source}. Each source is a pc.backtest (e.g. a value of
                 the `ports` dict from run_window / infer_window) OR a fitted model
                 carrying `.portfolio` (the net_roll_test workflow); both expose
                 .weights [periods x assets] and .dates.
    asset_labels : y-axis tick labels (e.g. Y.data.columns); default 0..n_y-1.
    names        : subset/order of labels to plot; default all keys of `portfolios`.
    ncols        : panels per row.
    figsize      : figure size; default scales with the grid.
    """
    names = list(portfolios) if names is None else names
    ncols = min(ncols, len(names))
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (7.5 * ncols, 3.2 * nrows),
                             squeeze=False)
    # A long-only solve (long_short=False) constrains z >= 0 only to ECOS tolerance
    # (strict ~1e-8/1e-9, relaxed retry 1e-6), so the returned weights carry tiny
    # negative slack (~ -1e-7) that is not a real short. Gate the diverging colormap
    # on a tolerance so that slack does not flip the meter into a symmetric ± scale;
    # sub-tolerance negatives are clipped to 0 and shown on the sequential scale.
    W_TOL = 1e-6
    for ax, name in zip(axes.flat, names):
        bt = getattr(portfolios[name], 'portfolio', portfolios[name])   # model -> its backtest
        w = np.asarray(bt.weights)
        n_periods, n_y = w.shape
        dates = getattr(bt, 'dates', None)
        if w.min() < -W_TOL:                                   # genuine shorts: diverging at 0
            vmax = np.abs(w).max()
            im = ax.imshow(w.T, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                           interpolation='nearest')
        else:                                                  # long-only: sequential from 0
            w = np.clip(w, 0.0, None)                          # drop sub-tolerance solver slack
            im = ax.imshow(w.T, aspect='auto', cmap='viridis', vmin=0, interpolation='nearest')
        if dates is not None:
            tick = np.linspace(0, n_periods - 1, min(6, n_periods), dtype=int)
            ax.set_xticks(tick)
            ax.set_xticklabels([dates[i].strftime('%Y-%m') for i in tick],
                               rotation=45, fontsize=7)
        ax.set_yticks(range(n_y))
        ax.set_yticklabels(list(asset_labels) if asset_labels is not None else range(n_y),
                           fontsize=6)
        ax.set_title(str(name), fontsize=10)
        fig.colorbar(im, ax=ax, label='weight')
    for ax in axes.flat[len(names):]:                          # hide unused grid cells
        ax.axis('off')
    fig.suptitle('Allocation weights over time', fontsize=13)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Interactive Plotly plotters
# ---------------------------------------------------------------------------
def _dash(ls):
    return 'dash' if ls == '--' else 'solid'


def plot_wealth(names, portfolios, title='All Models: Cumulative Wealth'):
    """Interactive cumulative wealth chart (Plotly). Legend placed outside right.

    The single wealth plotter for the whole notebook — pass any subset of `names`
    with a custom `title` for paired comparisons, or all eight for the overview.
    ROM models are drawn thicker so estimation-robust curves stand out.
    """
    dates = [d.strftime('%Y-%m-%d') for d in _wealth_anchor(portfolios[0].rets.index)]
    fig = go.Figure()
    for name, port in zip(names, portfolios):
        color, ls = COLORS[name]
        fig.add_trace(go.Scatter(
            x=dates, y=[1.0, *port.rets['tri'].values.tolist()],
            mode='lines', name=name,
            line=dict(color=color, dash=_dash(ls), width=2.5 if 'ROM' in name else 1.8)
        ))
    fig.update_layout(
        title=title,
        xaxis_title='Date', yaxis_title='Cumulative Wealth',
        hovermode='x unified',
        legend=dict(x=1.01, y=0.5, xanchor='left', yanchor='middle'),
        margin=dict(r=180),
        height=500
    )
    fig.show()


# plot_all_wealth is the historical name for the all-models overview; kept as an
# alias so the §4/§6 call sites read naturally.
plot_all_wealth = plot_wealth


def plot_drawdown(names, portfolios, title):
    """Interactive running drawdown chart (Plotly). Legend placed outside right."""
    dates = [d.strftime('%Y-%m-%d') for d in portfolios[0].rets.index]
    fig = go.Figure()
    for name, port in zip(names, portfolios):
        color, ls = COLORS[name]
        fig.add_trace(go.Scatter(
            x=dates, y=running_dd(port).values.tolist(),
            mode='lines', name=name,
            line=dict(color=color, dash=_dash(ls), width=1.5)
        ))
    fig.add_hline(y=0, line=dict(color='black', width=0.5))
    fig.update_layout(
        title=title,
        xaxis_title='Date', yaxis_title='Drawdown',
        hovermode='x unified',
        legend=dict(x=1.01, y=0.5, xanchor='left', yanchor='middle'),
        margin=dict(r=180),
        height=420
    )
    fig.show()


def plot_summary_bars(all_names, all_metrics):
    """Interactive horizontal bar chart — Sharpe and Effective Holdings (Plotly)."""
    bar_colors = [COLORS[n][0] for n in all_names]
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['Sharpe Ratio', 'Effective Holdings'])
    fig.add_trace(go.Bar(
        y=all_names, x=all_metrics['Sharpe'].values.tolist(),
        orientation='h', marker_color=bar_colors, showlegend=False,
        hovertemplate='%{y}: %{x:.3f}<extra></extra>'
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        y=all_names, x=all_metrics['Eff. Holdings'].values.tolist(),
        orientation='h', marker_color=bar_colors, showlegend=False,
        hovertemplate='%{y}: %{x:.2f}<extra></extra>'
    ), row=1, col=2)
    fig.update_layout(title='Performance Summary', hovermode='closest', height=420)
    fig.show()


def plot_epsilon_sweep(sweep, metric='Sharpe', title=None):
    """Line chart of a metric vs fixed ε, one trace per model (Plotly).

    Reads a run_epsilon_sweep result and overlays the estimation-robust models' ε-curves, so the
    fixed-B (PO-ROM) and trained-B (SPO-ROM) robustness dials are compared at a glance. ROM curves
    are drawn thicker, matching plot_wealth. See docs/BASE_ROM_LAYER.md "Long-panel investigation".

    sweep  : DataFrame from run_epsilon_sweep -- MultiIndex (model, epsilon), metric columns.
    metric : which build_metrics_table column to put on y (default 'Sharpe').
    """
    fig = go.Figure()
    for name in sweep.index.get_level_values('model').unique():
        sub = sweep.xs(name, level='model')
        color, ls = COLORS.get(name, ('black', '-'))
        fig.add_trace(go.Scatter(
            x=sub.index.tolist(), y=sub[metric].tolist(),
            mode='lines+markers', name=name,
            line=dict(color=color, dash=_dash(ls), width=2.5 if 'ROM' in name else 1.8)))
    fig.update_layout(
        title=title or f'ε sweep: {metric} vs fixed ε (walk-forward)',
        xaxis_title='ε (fixed)', yaxis_title=metric,
        hovermode='x unified',
        legend=dict(x=1.01, y=0.5, xanchor='left', yanchor='middle'),
        margin=dict(r=160), height=440)
    fig.show()


# ---------------------------------------------------------------------------
# Experiment helpers: model factory, single-window fit/infer, date windowing
# ---------------------------------------------------------------------------
# Anti-clutter principle: hyperparameters are bound to each model once, at
# build_models (cfg is stashed as model.cfg; e2e_net already stores n_obs,
# epochs, lr, ...). fit_window / infer_window / run_window therefore take bare
# signatures and read settings off the model.
#
# Correctness rule: fetch_data_from_disk returns X = X_raw[:-1], Y = Y_raw[1:],
# so X.data and Y.data carry different date labels but are positionally aligned
# (X.iloc[i] predicts Y.iloc[i], realized at Y.data.index[i]). date_window
# resolves boundaries on the Y.data timeline and slices BOTH frames by the same
# integer positions, reproducing the existing X / lagged-Y alignment.

_SPO_SPECS = [
    ('SPO-M',         dict(opt_layer='base_mod')),
    ('SPO-ROM',       dict(opt_layer='base_rom')),
    ('SPO-Nominal',   dict(opt_layer='nominal')),
    ('SPO-TV',        dict(opt_layer='tv')),
    ('SPO-Hellinger', dict(opt_layer='hellinger')),
]


def build_models(cfg, n_x, n_y, n_obs, which=None):
    """Instantiate the model zoo and bind cfg to each model.

    cfg     : dict of hyperparameters (seed, epochs, lr, weight_decay, dro_lr,
              max_weight, long_short, target_ratio, ...). Stashed as model.cfg.
    n_x     : number of features.
    n_y     : number of assets.
    n_obs   : rolling-window size.
    which   : optional iterable of model names to keep (default: all eight).
    returns : dict {name: model} with cfg attached to every model.
    """
    # Robust-solve config passed to the (owned) library solve strategy: per-opt_layer strict
    # settings + relaxed retry + equal-weight fallback (replaces the old forward monkeypatch).
    _retry = dict(solve_retry_args=_RETRY_ARGS, solve_fallback=True)

    # Annualization factor: explicit cfg['periods_per_year'] wins, else derive from cfg['freq'].
    pp = cfg.get('periods_per_year') or pc.PERIODS_PER_YEAR.get(cfg.get('freq', 'weekly'), 52)

    ew = bm.equal_weight(n_x=n_x, n_y=n_y, n_obs=n_obs, periods_per_year=pp)
    po_m = bm.pred_then_opt(n_x, n_y, n_obs, opt_layer='base_mod',
                            max_weight=cfg['max_weight'], long_short=cfg['long_short'],
                            periods_per_year=pp,
                            set_seed=cfg['seed'], solver_args=SOLVER_ARGS['base_mod'], **_retry).double()
    po_m._opt_layer = 'base_mod'
    po_rom = bm.pred_then_opt(n_x, n_y, n_obs, opt_layer='base_rom', epsilon=0.5,
                              max_weight=cfg['max_weight'], long_short=cfg['long_short'],
                              periods_per_year=pp,
                              set_seed=cfg['seed'], solver_args=SOLVER_ARGS['base_rom'], **_retry).double()
    po_rom._opt_layer = 'base_rom'

    shared = dict(n_x=n_x, n_y=n_y, n_obs=n_obs, pred_model='linear',
                  epochs=cfg['epochs'], lr=cfg['lr'], weight_decay=cfg['weight_decay'],
                  max_weight=cfg['max_weight'], long_short=cfg['long_short'],
                  periods_per_year=pp,
                  set_seed=cfg['seed'])
    models = {'EW': ew, 'PO-M': po_m, 'PO-ROM': po_rom}
    for name, spec in _SPO_SPECS:
        kw = dict(spec)
        if kw['opt_layer'] in ('nominal', 'tv', 'hellinger'):
            kw['dro_lr'] = cfg['dro_lr']
        if kw['opt_layer'] == 'base_rom':
            # epsilon is passed at instantiation (CV-selected via cfg['epsilon']; default 0.5 to
            # match PO-ROM). It is frozen -- not gradient-learned -- per e2e_net defaults.
            kw['epsilon'] = cfg.get('epsilon', 0.5)
        m = e2e_net(**shared, **kw, solver_args=SOLVER_ARGS[kw['opt_layer']], **_retry).double()
        m._opt_layer = kw['opt_layer']
        # _init_state (the in-memory pristine snapshot fit_window resets from) is now created by
        # e2e_net.__init__ itself -- per-object, so tv/hellinger can't alias each other.
        models[name] = m

    for name, m in models.items():
        m.cfg = cfg
        m._name = name          # labels trained_params / model_report rows
    if which is not None:
        models = {k: models[k] for k in which}
    return models


def calibrate_models(models, X_train, Y_train, target_ratio=None):
    """One-time calibration of pred_loss_factor for every SPO (e2e_net) model.

    pred_loss_factor must be calibrated *at the OLS initialization* (see CLAUDE.md
    and e2e_net.calibrate_pred_loss_factor), so each model is OLS warm-started here
    — with the same fit_predictor (OLS + Sigma_mu_hat) that fit_window applies — before
    the calibration forward pass. The OLS init is deterministic in (X_train, Y_train),
    so the later fit_window(reset=True) re-derives the identical weights; pred_loss_factor
    is a plain attribute and survives that reset.

    models       : dict {name: model} from build_models.
    X_train      : standardized train-slice feature DataFrame from date_window
                   (the same features fit_window trains on — do NOT pass raw
                   load_data output, or calibration scale won't match the fit).
    Y_train      : training-window return DataFrame.
    target_ratio : override; default uses each model's cfg['target_ratio'].
    returns      : dict {name: calibrated pred_loss_factor}.
    """
    out = {}
    for name, m in models.items():
        if isinstance(m, e2e_net) and m.pred_loss is not None:
            tr = target_ratio if target_ratio is not None else m.cfg.get('target_ratio', 0.5)
            m.fit_predictor(X_train, Y_train)   # OLS warm-start + (base_rom) Sigma_mu_hat build
            m._solve_phase = 'calibrate'
            out[name] = m.calibrate_pred_loss_factor(X_train, Y_train, tr)
            rec = getattr(m, '_recorder', None)
            if rec is not None:
                rec.record_meta(m, pred_loss_factor=out[name], target_ratio=tr,
                                weight_decay=m.weight_decay)
    return out


def _standardize(X_train_df, *others):
    """Z-score features on X_train_df's stats; apply to it and each extra frame.

    Fit on the train slice only (no look-ahead); pass test slices as *others to
    scale them by the same train stats. Inputs must be RAW (load_data is raw).
    """
    mu = X_train_df.mean()
    sigma = X_train_df.std().replace(0.0, 1.0)   # guard a constant feature
    z = lambda d: (d - mu) / sigma
    return z(X_train_df) if not others else (z(X_train_df), *(z(o) for o in others))


def _stash_fit_attrs(model, X_train_df, Y_train_df, Theta):
    """Attach labeled regression params / residuals to a fitted model.

    Sets model.B_, .alpha_, .resid_, .y_hat_, .theta_l2_, .theta_dist_l2_.
    """
    assets = list(Y_train_df.columns)
    factors = list(X_train_df.columns)
    W = model.pred_layer.weight.detach().cpu().numpy()
    b = model.pred_layer.bias.detach().cpu().numpy()
    Yhat = X_train_df.values @ W.T + b
    model.B_ = pd.DataFrame(W, index=assets, columns=factors)
    model.alpha_ = pd.Series(b, index=assets, name='alpha')
    model.resid_ = pd.DataFrame(Y_train_df.values - Yhat, index=Y_train_df.index, columns=assets)
    model.y_hat_ = pd.Series(Yhat[-1], index=assets, name='y_hat')
    w = model.pred_layer.weight.detach().cpu()
    bb = model.pred_layer.bias.detach().cpu()
    Theta_cpu = Theta.detach().cpu()
    # True L2 magnitudes (norm of theta; L2 distance from the OLS warm-start), matching the
    # per-epoch telemetry in e2e_net._emit_epoch -- the `_l2` name is the norm, not its square.
    model.theta_l2_ = float(torch.sqrt(torch.sum(w ** 2) + torch.sum(bb ** 2)).item())
    model.theta_dist_l2_ = float(torch.sqrt(torch.sum((w - Theta_cpu[:, 1:]) ** 2)
                                            + torch.sum((bb - Theta_cpu[:, 0]) ** 2)).item())


def fit_window(model, X_train_df, Y_train_df, *, reset=True):
    """Train a model on a single explicit window (no rolling, no net_roll_test).

    model       : equal_weight | pred_then_opt | e2e_net.
    X_train_df  : training-window feature DataFrame.
    Y_train_df  : training-window return DataFrame, aligned with X_train_df.
    reset       : if True, restore e2e_net learnable params from the in-memory
                  pristine snapshot (model._init_state, taken in build_models)
                  before training (ignored for the other model types). Matters
                  only when the same model object is refit across windows.
    returns     : the (fitted) model, with introspection attributes attached.
    """
    if isinstance(model, bm.equal_weight):
        return model

    if isinstance(model, bm.pred_then_opt):
        Theta = model.fit_predictor(X_train_df, Y_train_df)   # OLS + (base_rom) Sigma_mu_hat
        _stash_fit_attrs(model, X_train_df, Y_train_df, Theta)
        return model

    # e2e_net
    if reset and hasattr(model, '_init_state'):
        model.load_state_dict(model._init_state)
    Theta = model.fit_predictor(X_train_df, Y_train_df)       # OLS + (base_rom) Sigma_mu_hat
    train_set = DataLoader(pc.SlidingWindow(X_train_df, Y_train_df, model.n_obs, model.perf_period))
    model._solve_phase = 'train'
    model._solve_window = 0          # single-window run: all solves/epochs key to window 0
    model.net_train(train_set)
    _stash_fit_attrs(model, X_train_df, Y_train_df, Theta)
    return model


def infer_window(model, X_df, Y_df):
    """Run inference over a window and return a portfolio of forward returns.

    The frame's first n_obs rows are lookback; predictions are emitted for every
    valid date Y_df.index[n_obs:] that the sliding window admits. Each step
    records the single realized next-period return (the sliding window's forward
    horizon is fixed at 1 — inference needs exactly one return per decision date,
    matching date_window's trailing buffer; it is not a user choice).

    model   : a fitted model (equal_weight uses 1/n weights).
    X_df    : feature DataFrame covering [pred_start - n_obs ... pred_end].
    Y_df    : return DataFrame, positionally aligned with X_df.
    returns : pc.backtest with weights, rets, dates and computed stats().
    """
    n_obs = model.n_obs
    ds = pc.SlidingWindow(X_df, Y_df, n_obs, 1)
    is_ew = isinstance(model, bm.equal_weight) or not hasattr(model, 'forward')
    device = next(model.parameters()).device if not is_ew else None
    if not is_ew:
        model._solve_phase = 'infer'
        model._solve_window = 0

    weights, rets, dates = [], [], []
    with torch.no_grad():
        for j in range(len(ds)):
            x, y, y_perf = ds[j]
            if is_ew:
                w = np.ones(model.n_y) / model.n_y
            else:
                model._solve_date = Y_df.index[j + n_obs]
                z, _ = model(x.to(device), y.to(device))
                w = z.squeeze().cpu().numpy()
            weights.append(w)
            rets.append(float(y_perf[0].cpu().numpy() @ w))
            dates.append(Y_df.index[j + n_obs])

    dates = pd.DatetimeIndex(dates)
    port = pc.backtest(len(weights), model.n_y, dates,
                       periods_per_year=getattr(model, 'periods_per_year', 52))
    port.weights = np.asarray(weights)
    port.rets = np.asarray(rets)
    port.dates = dates
    port.stats()
    return port


def date_window(X, Y, train_end=None, pred_start=None, pred_end=None):
    """Slice train / test DataFrames by date on the Y (realized-return) timeline.

    Both X.data and Y.data are sliced by the SAME integer positions so the
    existing X / lagged-Y alignment is preserved. The test slice prepends n_obs
    lookback rows and is sized (via n_pred) so infer_window emits predictions for
    exactly the [pred_start, pred_end] range.

    X, Y       : TrainTest objects from fetch_data_from_disk.
    train_end  : last training date (inclusive). Default: just before pred_start.
    pred_start : first prediction date. Default: first date after train_end.
    pred_end   : last prediction date. Default: end of data (last date dropped,
                 since one forward row is required).
    returns    : (X_train_df, Y_train_df, X_test_df, Y_test_df).
    """
    idx = Y.data.index
    n_obs = X.n_obs
    n = len(idx)

    if train_end is not None:
        t_end = int(idx.searchsorted(pd.Timestamp(train_end), side='right')) - 1
    elif pred_start is not None:
        t_end = int(idx.searchsorted(pd.Timestamp(pred_start), side='left')) - 1
    else:
        raise ValueError("Provide at least one of train_end or pred_start.")

    p_start = (int(idx.searchsorted(pd.Timestamp(pred_start), side='left'))
               if pred_start is not None else t_end + 1)
    p_end = (int(idx.searchsorted(pd.Timestamp(pred_end), side='right')) - 1
             if pred_end is not None else n - 1)

    a = p_start - n_obs
    if a < 0:
        raise ValueError(
            f"Insufficient lookback: pred_start position {p_start} < n_obs={n_obs}. "
            f"Move pred_start at least {n_obs} observations into the data."
        )
    # Derive the frame end from the number of decisions we want, so the SlidingWindow emits
    # predictions for exactly [pred_start, pred_end] and no magic buffer is needed:
    #   len(SlidingWindow(frame, n_obs, perf_period=1)) == frame_len - n_obs == n_pred.
    n_pred = p_end - p_start + 1                 # decisions wanted (inclusive)
    b = min(a + n_obs + n_pred, n)               # frame length so len(SlidingWindow)==n_pred

    X_train_df = X.data.iloc[:t_end + 1]
    Y_train_df = Y.data.iloc[:t_end + 1]
    X_test_df = X.data.iloc[a:b]
    Y_test_df = Y.data.iloc[a:b]

    # Machine-checked invariant: the test frame yields exactly the intended decision count.
    # Guards against date_window and SlidingWindow.__len__ silently drifting apart again.
    assert len(pc.SlidingWindow(X_test_df, Y_test_df, n_obs, 1)) == n_pred, (
        f"date_window/SlidingWindow mismatch: got "
        f"{len(pc.SlidingWindow(X_test_df, Y_test_df, n_obs, 1))} windows, expected {n_pred}"
    )

    # Standardize on this window's train slice; apply to train + test (incl. the
    # n_obs lookback rows in X_test_df). The test stats come from the past only.
    X_train_df, X_test_df = _standardize(X_train_df, X_test_df)
    return X_train_df, Y_train_df, X_test_df, Y_test_df


def run_window(models, X, Y, train_end, pred_start=None, pred_end=None, report=True):
    """Fit every model on the train slice and infer on the holdout slice.

    models     : dict {name: model} from build_models (already calibrated).
    X, Y       : TrainTest objects from fetch_data_from_disk.
    train_end  : last training date (inclusive).
    pred_start : first prediction date (default: just after train_end).
    pred_end   : last prediction date (default: end of data).
    report     : if True, print the solver retry/fallback summary after fitting.
    returns    : dict {name: pc.backtest} of out-of-sample portfolios.
    """
    Xtr, Ytr, Xte, Yte = date_window(X, Y, train_end, pred_start, pred_end)
    ports = {}
    for name, m in models.items():
        fit_window(m, Xtr, Ytr)
        ports[name] = infer_window(m, Xte, Yte)
        rec = getattr(m, '_recorder', None)
        if rec is not None:                     # single-window boundary -> lazy diagnostics
            rec.on_window(m, 0, ports[name].dates, Xtr, Ytr)
    if report:
        rep = solve_report(models)
        active = rep[(rep != 0).any(axis=1)]
        if len(active):
            print('Solver retries / fallbacks (windows below strict tolerance):')
            print(active.to_string())
        else:
            print('Solver: all windows reached strict tolerance (no retries/fallbacks).')
    return ports


def run_feature_sweep(cfg, variants, X_full, Y, *, train_end, pred_start=None,
                      pred_end=None, which=None, attach_report=True, progress=True):
    """Single-window ablation over feature subsets of an already-loaded panel.

    For each {label: features} in variants, subset the columns of X_full (no disk
    re-read) and run the standard single-window cycle on that subset: date_window ->
    build_models -> calibrate_models -> run_window. The full panel is loaded once by
    the caller and reused, so the sweep itself parses no data.

    cfg           : hyperparameter dict (as build_models expects).
    variants      : dict {label: features}; features is a group/name selection understood
                    by DataLoad (e.g. 'ff5+mom', ['ff5+mom', 'esg'], None for all 12).
    X_full, Y     : TrainTest objects from load_data with the FULL feature set.
    train_end     : last training date (inclusive); pred_start / pred_end as in run_window.
    which         : optional subset of model names to build (default: all eight).
    attach_report : if True, attach a fresh RunReport to each subset before calibrating,
                    so its meta / solves / diagnostics are captured.
    returns       : dict {label: {'X', 'models', 'ports', 'metrics', 'report'}}.
    """
    n_obs, n_y = X_full.n_obs, Y.train().shape[1]
    panel_cols = list(X_full.data.columns)             # resolve against the ACTIVE panel, so a
    out = {}                                           # non-ESG panel (market) resolves its own names
    var_bar = track(list(variants.items()), total=len(variants), desc='feature-sweep',
                    enable=progress)
    for label, features in var_bar:
        var_bar.set_postfix(subset=label)
        cols = dl._resolve_features(features, panel_cols)   # single source of truth for the mapping
        X = dl.TrainTest(X_full.data[cols], n_obs, X_full.split)   # column subset keeps the row lag
        Xtr, Ytr, _, _ = date_window(X, Y, train_end, pred_start, pred_end)
        models = build_models(cfg, len(cols), n_y, n_obs, which)
        report = RunReport().attach(models) if attach_report else None
        calibrate_models(models, Xtr, Ytr)             # standardized Xtr matches the fit scale
        ports = run_window(models, X, Y, train_end, pred_start, pred_end, report=False)
        metrics = build_metrics_table(list(ports), list(ports.values()))
        out[label] = {'X': X, 'models': models, 'ports': ports,
                      'metrics': metrics, 'report': report}
    return out


def run_epsilon_sweep(cfg, X, Y, epsilons, *, n_roll=8, models=('PO-ROM', 'SPO-ROM'),
                      train_end=None, pred_start=None, pred_end=None, progress=True):
    """Walk-forward epsilon sweep for the estimation-robust models -- a DIAGNOSTIC ONLY.

    For each (model, eps) it pins eps, calibrates, and backtests walk-forward, so every eps yields
    the fixed-B (PO-ROM, OLS per window) and trained-B (SPO-ROM, end-to-end per window) curve side
    by side. Use it to see the SHAPE of the eps->performance response (flat = robust, spiked =
    fragile), NOT to pick eps: epsilon is selected data-driven by cross-validation
    (e2e_net.net_cv with an 'epsilon' grid on the training folds). It is NOT gradient-learned --
    the epsilon gradient is empirically uninformative, so epsilon is frozen during net_train.

    IMPORTANT: if you read this curve to inform an epsilon choice, run it on a window DISJOINT from
    the reported holdout (pass an earlier train_end / pred bounds), otherwise you are choosing on
    the period you report -- the same leakage the CV selection avoids.

    cfg        : hyperparameter dict (as build_models expects).
    X, Y       : TrainTest objects. For rolling, their split defines the walk-forward holdout.
    epsilons   : iterable of fixed epsilon values to sweep.
    n_roll     : rolling windows for net_roll_test (the deployment eval, B refreshed per window).
                 n_roll=None -> one single-window run_window (quick look; a stale B biases the
                 curve toward higher eps, so it is NOT the deployment case -- needs train_end).
    models     : which base_rom models to sweep (default both PO-ROM and SPO-ROM).
    train_end / pred_start / pred_end : single-window bounds (only used when n_roll is None).
    returns    : DataFrame, MultiIndex (model, epsilon), columns = build_metrics_table metrics.
    """
    n_x, n_y, n_obs = X.train().shape[1], Y.train().shape[1], X.n_obs
    if n_roll is not None:                              # calibrate on the split's initial train
        Xtr_raw = X.train()                            # window (== roll window 0), standardized on
        mu, sig = Xtr_raw.mean(), Xtr_raw.std().replace(0.0, 1.0)   # its own stats -- matches the
        Xtr, Ytr = (Xtr_raw - mu) / sig, Y.train()     # per-window scaling net_roll_test applies
    elif train_end is not None:
        Xtr, Ytr, _, _ = date_window(X, Y, train_end, pred_start, pred_end)
    else:
        raise ValueError("single-window mode (n_roll=None) needs train_end.")

    rows = {}
    combos = [(name, v) for name in models for v in epsilons]
    sweep_bar = track(combos, total=len(combos), desc='ε-sweep', enable=progress)
    for name, v in sweep_bar:
        sweep_bar.set_postfix(model=name, eps=v)
        built = build_models(cfg, n_x, n_y, n_obs, which=[name])
        m = built[name]
        if getattr(m, 'model_type', None) != 'base_rom':
            raise ValueError(f"run_epsilon_sweep: '{name}' is not a base_rom model.")
        m.epsilon.data.fill_(float(v))                 # pin epsilon (both models read self.epsilon)
        if isinstance(m, e2e_net):                     # SPO: freeze so training can't move it ...
            m.epsilon.requires_grad_(False)
            m._init_state = copy.deepcopy(m.state_dict())   # ... and per-roll resets keep v
        calibrate_models(built, Xtr, Ytr)              # SPO only; PO has no pred_loss (skipped)
        if n_roll is not None:
            m.net_roll_test(X, Y, n_roll=n_roll, progress=progress)
            port = m.portfolio
        else:
            port = run_window(built, X, Y, train_end, pred_start, pred_end, report=False)[name]
        rows[(name, float(v))] = build_metrics_table([name], [port]).iloc[0]

    out = pd.DataFrame(rows).T
    out.index.set_names(['model', 'epsilon'], inplace=True)
    return out


# ---------------------------------------------------------------------------
# Feature-relation diagnostics: pick the feature subset the backtest runs on by
# looking at how the features relate to each other (collinearity) and to the
# target (signal), then confirming the choice against the ablation sweep.
# ---------------------------------------------------------------------------
def feature_variants(X):
    """Panel-adaptive menu of nested feature subsets for the ablation sweep.

    ESG disk panel -> the semantic groups (FF5+MOM / +Macro / +ESG / All-12).
    Any other panel (e.g. the 8-factor market panel) -> nested subsets taken from
    the panel's own column order, so the sweep still runs without ESG group names.
    For the FF market panel the count breakpoints line up with FF3 / FF5 / FF5+MOM
    / All; the labels report the feature count to stay honest about arbitrary panels.

    X       : TrainTest with the full feature panel.
    returns : dict {label: features} understood by run_feature_sweep (a group-name
              selection for the ESG panel, else an explicit column list per subset).
    """
    cols = list(X.data.columns)
    group_cols = {c for g in dl.FEATURE_GROUPS.values() for c in g}
    if set(cols) == group_cols:                    # the ESG disk panel: use semantics
        return {'FF5+MOM': 'ff5+mom',
                '+Macro':  ['ff5+mom', 'macro'],
                '+ESG':    ['ff5+mom', 'esg'],
                'All-12':  None}
    # Generic panel: nested prefixes at sensible breakpoints (FF3/FF5/FF5+MOM/All
    # when the columns are the ordered FF factors).
    n = len(cols)
    breaks = sorted({k for k in (3, 5, 6, n) if 1 <= k <= n})
    return {f'{k}-feat': cols[:k] for k in breaks}


def _vif(Xz):
    """Variance Inflation Factor per feature: VIF_i = 1 / (1 - R_i^2), where R_i^2 is from
    regressing feature i on all the other features.

    Returns np.inf where a feature is (near-)perfectly explained by the rest (a singular
    correlation block, true VIF -> inf) -- the honest limit, rather than the small pseudo-inverse
    diagonal that a floored 1/(1-R^2) would mask as "no collinearity". With p features this is p
    tiny least-squares solves; for a full-rank block it equals the inverse-correlation diagonal.
    """
    A = Xz.values
    n, p = A.shape
    vifs = []
    for i in range(p):
        others = np.column_stack([np.ones(n), np.delete(A, i, axis=1)])   # intercept + the rest
        beta, *_ = np.linalg.lstsq(others, A[:, i], rcond=None)
        ss_res = float(((A[:, i] - others @ beta) ** 2).sum())
        ss_tot = float(((A[:, i] - A[:, i].mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vifs.append(np.inf if r2 >= 1 - 1e-10 else 1.0 / (1.0 - r2))
    return pd.Series(vifs, index=list(Xz.columns))


def feature_diagnostics(X, Y, train_end, show=True):
    """Variable-relation diagnostic on the train slice: feature collinearity.

    Computed on the standardized training window only (via date_window), so there is no
    look-ahead. `vif` is the Variance Inflation Factor per feature (np.inf when a feature is
    (near-)perfectly collinear with the others); `max_abs_corr_other` is each feature's strongest
    pairwise correlation with another feature. Both are redundancy measures (lower = more
    independent); relevance is owned by the ablation sweep. See docs/DIAGNOSTICS_DEFINITIONS.md.

    X, Y      : TrainTest objects from load_data.
    train_end : last training date (inclusive) -- the window the diagnostic reads.
    show      : if True, render the feature-correlation heatmap (Plotly).
    returns   : DataFrame indexed by feature with columns ['vif', 'max_abs_corr_other'].
    """
    Xtr, _, _, _ = date_window(X, Y, train_end=train_end)
    corr = Xtr.corr()
    off_diag = corr.where(~np.eye(len(corr), dtype=bool))
    diag = pd.DataFrame({
        'vif': _vif(Xtr),
        'max_abs_corr_other': off_diag.abs().max(),
    }, index=corr.columns).round(3)

    if show:
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
            zmin=-1, zmax=1, colorscale='RdBu_r', reversescale=False,
            colorbar=dict(title='corr')))
        fig.update_layout(title='Feature correlation (train slice)',
                          height=480, width=560, yaxis=dict(autorange='reversed'))
        fig.show()
    return diag


def cv_score_subset(cfg, X_sub, Y, *, n_val=4, which='SPO-Nominal'):
    """Averaged net_cv validation-fold loss for one feature subset (train-only; no report leak).

    Builds the representative model on the subset and runs net_cv's expanding folds at the current
    cfg (a 1-point grid, so nothing is swept -- we just read the averaged fold val_loss). net_cv
    operates on X_sub.train() only, so the report holdout is never touched. Lower is better.

    cfg   : hyperparameter dict (as build_models expects).
    X_sub : TrainTest whose columns are already the subset (report split preserved).
    Y     : TrainTest of returns.
    n_val : number of expanding validation folds.
    which : representative model name used to score relevance (default SPO-Nominal).
    """
    n_x, n_y, n_obs = X_sub.train().shape[1], Y.train().shape[1], X_sub.n_obs
    m = build_models(cfg, n_x, n_y, n_obs, which=[which])[which]
    m.net_cv(X_sub, Y, grid={'epochs': [cfg['epochs']]}, n_val=n_val)
    return float(m.cv_results['val_loss'].iloc[0])


def select_best_subset(diag, variants, cfg, X_full, Y, *, n_val=4, vif_threshold=10.0,
                       which='SPO-Nominal'):
    """Choose the feature subset for the rolling backtest -- WITHOUT touching the report holdout.

    Prune features the diagnostic flags as collinear (VIF > vif_threshold), then among subsets that
    use only surviving features pick the one with the best (lowest) net_cv validation-fold loss on
    the training data (falls back to all subsets if pruning leaves none fully intact). Selection
    never sees the report period -- that leakage is why the old best-OOS-Sharpe tie-break, which
    ranked by Sharpe on the reported holdout, was removed.

    diag          : per-feature frame from feature_diagnostics.
    variants      : dict {label: features} (as run_feature_sweep expects).
    cfg           : hyperparameter dict.
    X_full, Y     : TrainTest objects with the FULL feature panel (report split preserved).
    n_val         : expanding validation folds used to score each subset.
    vif_threshold : VIF above which a feature is treated as redundant.
    which         : representative model used to score relevance.
    returns       : (best_label, best_cols) -- the winning subset's columns.
    """
    survivors = set(diag.index[diag['vif'] <= vif_threshold])
    panel_cols = list(X_full.data.columns)
    n_obs = X_full.n_obs
    scored = []
    for label, features in variants.items():
        cols = dl._resolve_features(features, panel_cols)
        X_sub = dl.TrainTest(X_full.data[cols], n_obs, X_full.split)   # column subset keeps row lag
        val = cv_score_subset(cfg, X_sub, Y, n_val=n_val, which=which)
        scored.append((label, cols, val, set(cols) <= survivors))
    pool = [s for s in scored if s[3]] or scored          # prefer prune-surviving subsets
    best_label, best_cols, _, _ = min(pool, key=lambda s: s[2])   # lowest validation loss
    best_cols = [c for c in best_cols if c in survivors] or best_cols
    return best_label, best_cols


def trained_params(model):
    """One-row view of a model's learned scalars and prediction-weight norms.

    model  : a fitted model.
    returns: pd.Series (gamma, delta, epsilon, pred_loss_factor, theta_l2,
             theta_dist_l2; NaN where not applicable).
    """
    def _scalar(attr):
        v = getattr(model, attr, None)
        if v is None:
            return np.nan
        try:
            return float(v.item())
        except (AttributeError, ValueError, TypeError):
            try:
                return float(v)
            except (ValueError, TypeError):
                return np.nan

    plf = getattr(model, 'pred_loss_factor', None)
    # gamma enters the objective only for the nominal / DRO layers; base_mod / base_rom declare it
    # but never use it, so report NaN there rather than the inert init draw (see e2edro.base_mod).
    uses_gamma = getattr(model, 'model_type', None) in ('nom', 'dro')
    return pd.Series({
        'model_type': getattr(model, 'model_type', 'ew'),
        'gamma': _scalar('gamma') if uses_gamma else np.nan,
        'delta': _scalar('delta'),
        'epsilon': _scalar('epsilon'),
        'pred_loss_factor': float(plf) if isinstance(plf, (int, float)) else np.nan,
        'theta_l2': getattr(model, 'theta_l2_', np.nan),
        'theta_dist_l2': getattr(model, 'theta_dist_l2_', np.nan),
    }, name=getattr(model, '_name', None))


def model_report(models):
    """Stack trained_params across a dict of models into one table.

    models : dict {name: model}.
    returns: pd.DataFrame indexed by model name.
    """
    return pd.DataFrame({name: trained_params(m) for name, m in models.items()}).T


def roll_param_trajectory(report, names=None):
    """Per-window learning summary: learned params + all end-of-window diagnostics.

    Rolling analog of model_report (which reports single-window scalars), but drawn from the full
    per-epoch telemetry in `report.learning()`. For each (model, roll window) it keeps the LAST
    epoch -- the end-of-training state for that window -- so one row carries the learned robustness
    params (ε/γ/δ) alongside the complete learning diagnostics (loss_total/task/pred/val, grad-norm
    pred/robust, decay_norm, theta_l2, theta_dist_l2) and the epoch count. This is the "full
    picture" per window; the within-window per-epoch dynamics remain in report.learning(). Models
    with no training telemetry (EW, PO-*) do not appear. See docs/DIAGNOSTICS_DEFINITIONS.md.

    report : RunReport attached to the rolling run (net_roll_test).
    names  : subset/order of model names; default all trained models, in first-appearance order.
    returns: tidy DataFrame indexed by (model, window), one row per trained (model, window).
    """
    learn = report.learning()
    if learn.empty:
        return pd.DataFrame()
    last = (learn.sort_values('epoch')                      # end-of-training state per window
                 .groupby(['model', 'window'], as_index=False).last())
    n_ep = (learn.groupby(['model', 'window']).size()
                 .rename('n_epochs').reset_index())
    out = last.merge(n_ep, on=['model', 'window'])
    cols = ['model', 'window', 'epsilon', 'gamma', 'delta', 'n_epochs',
            'loss_total', 'loss_task', 'loss_pred', 'loss_val',
            'grad_norm_pred', 'grad_norm_robust', 'decay_norm', 'theta_l2', 'theta_dist_l2']
    out = out[[c for c in cols if c in out.columns]]

    if names is None:
        names = list(dict.fromkeys(learn['model']))         # first-appearance (run) order
    out = out[out['model'].isin(names)].copy()
    out['model'] = pd.Categorical(out['model'], categories=names, ordered=True)
    num = out.select_dtypes('number').columns
    out[num] = out[num].round(4)
    return out.sort_values(['model', 'window']).set_index(['model', 'window'])
