"""
Shared experiment + plotting harness for the ilo-portfolio-allocation notebooks
(base_rom_demo, esg_experiment_*, esg_alloc_*, e2e_ro_vs_dro_*, esg_diagnostic).

Contents
- Data loading (load_data) and metrics (portfolio_metrics, build_metrics_table,
  running_dd).
- Two plot families: matplotlib (plot_wealth, plot_epsilon_trajectory,
  plot_weight_heatmap) and Plotly (plot_all_wealth, plot_drawdown,
  plot_summary_bars). The `_px` names are backward-compat aliases for the Plotly
  plotters: esg_alloc_1..4 import them as `plot_all_wealth_px as plot_all_wealth`.
- Experiment helpers (build_models, calibrate_models, fit_window, infer_window,
  date_window, run_window, trained_params, model_report) that abstract model-zoo
  instantiation and the decoupled fit/inference flow out of the notebook cells.

Import patterns in use: `from base_rom_demo_utils import *` (pulls in np, pd,
torch, dl + every helper, keeping setup cells short), explicit
`from base_rom_demo_utils import (...)`, and the `_px`-aliased explicit import.
"""
import os as _os

# CUDA-deterministic reductions must be configured before torch initializes CUDA.
# The DRO cone programs (esp. TV) are ill-conditioned enough that the ~1e-7 GPU
# reduction noise amplifies to full-cap weight swings; together with the per-layer
# solver args + reject-inaccurate wrapper below and use_deterministic_algorithms()
# in set_seeds(), this makes same-seed/same-data runs bit-reproducible
# (see spo-critical-review.md Part II).
_os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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
    'plot_all_wealth', 'plot_drawdown', 'plot_summary_bars',
    'build_models', 'calibrate_models', 'fit_window', 'infer_window',
    'date_window', 'run_window', 'trained_params', 'model_report',
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
    ann = np.sqrt(52)
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
        'Ann. Turnover'  : round(p.turnover * 52, 2),
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


def plot_wealth(names, portfolios, title, figsize=(11, 5)):
    """Cumulative wealth curves for a list of (name, portfolio) pairs."""
    dates = _wealth_anchor(portfolios[0].rets.index)
    fig, ax = plt.subplots(figsize=figsize)
    for name, port in zip(names, portfolios):
        color, ls = COLORS[name]
        tri = np.concatenate([[1.0], port.rets['tri'].values])
        ax.plot(dates, tri,
                color=color, linestyle=ls, linewidth=2, label=name)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Wealth')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()


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
    for ax, name in zip(axes.flat, names):
        bt = getattr(portfolios[name], 'portfolio', portfolios[name])   # model -> its backtest
        w = np.asarray(bt.weights)
        n_periods, n_y = w.shape
        dates = getattr(bt, 'dates', None)
        if w.min() < 0:                                        # long-short: diverging at 0
            vmax = np.abs(w).max()
            im = ax.imshow(w.T, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                           interpolation='nearest')
        else:                                                  # long-only: sequential from 0
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


def plot_all_wealth(all_names, all_ports):
    """Interactive cumulative wealth chart (Plotly). Legend placed outside right."""
    dates = [d.strftime('%Y-%m-%d') for d in _wealth_anchor(all_ports[0].rets.index)]
    fig = go.Figure()
    for name, port in zip(all_names, all_ports):
        color, ls = COLORS[name]
        fig.add_trace(go.Scatter(
            x=dates, y=[1.0, *port.rets['tri'].values.tolist()],
            mode='lines', name=name,
            line=dict(color=color, dash=_dash(ls), width=2.5 if 'ROM' in name else 1.8)
        ))
    fig.update_layout(
        title='All Models: Cumulative Wealth',
        xaxis_title='Date', yaxis_title='Cumulative Wealth',
        hovermode='x unified',
        legend=dict(x=1.01, y=0.5, xanchor='left', yanchor='middle'),
        margin=dict(r=180),
        height=500
    )
    fig.show()


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


# Backward-compat aliases for the other notebooks that import the `_px` names
# (e.g. `from base_rom_demo_utils import plot_all_wealth_px`). Kept out of __all__
# so `import *` stays clean.
plot_all_wealth_px = plot_all_wealth
plot_drawdown_px = plot_drawdown
plot_summary_bars_px = plot_summary_bars


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

    cfg     : dict of hyperparameters (seed, epochs, lr, weight_decay, gamma_lr,
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

    ew = bm.equal_weight(n_x=n_x, n_y=n_y, n_obs=n_obs)
    po_m = bm.pred_then_opt(n_x, n_y, n_obs, opt_layer='base_mod',
                            max_weight=cfg['max_weight'], long_short=cfg['long_short'],
                            set_seed=cfg['seed'], solver_args=SOLVER_ARGS['base_mod'], **_retry).double()
    po_m._opt_layer = 'base_mod'
    po_rom = bm.pred_then_opt(n_x, n_y, n_obs, opt_layer='base_rom', epsilon=0.5,
                              max_weight=cfg['max_weight'], long_short=cfg['long_short'],
                              set_seed=cfg['seed'], solver_args=SOLVER_ARGS['base_rom'], **_retry).double()
    po_rom._opt_layer = 'base_rom'

    shared = dict(n_x=n_x, n_y=n_y, n_obs=n_obs, pred_model='linear',
                  epochs=cfg['epochs'], lr=cfg['lr'], weight_decay=cfg['weight_decay'],
                  max_weight=cfg['max_weight'], long_short=cfg['long_short'],
                  set_seed=cfg['seed'])
    models = {'EW': ew, 'PO-M': po_m, 'PO-ROM': po_rom}
    for name, spec in _SPO_SPECS:
        kw = dict(spec)
        if kw['opt_layer'] in ('nominal', 'tv', 'hellinger'):
            kw['gamma_lr'] = cfg['gamma_lr']
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
    model.theta_l2_ = float((torch.sum(w ** 2) + torch.sum(bb ** 2)).item())
    model.theta_dist_l2_ = float((torch.sum((w - Theta_cpu[:, 1:]) ** 2)
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
    port = pc.backtest(len(weights), model.n_y, dates)
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
    return pd.Series({
        'model_type': getattr(model, 'model_type', 'ew'),
        'gamma': _scalar('gamma'),
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
