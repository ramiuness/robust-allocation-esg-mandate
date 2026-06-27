"""
Utilities for base_rom_demo.ipynb.
Metric computation, running drawdown, and all plot functions (interactive Plotly).
The _px names are kept as backward-compat aliases for the other notebooks that
import them (see the bottom of the plot section).

Experiment helpers (build_models, calibrate_models, fit_window, infer_window,
date_window, run_window, trained_params, model_report) abstract the model-zoo
instantiation and the decoupled fit/inference flow out of the notebook cells.

Notebook prelude: a single `from base_rom_demo_utils import *` pulls in np, pd,
torch, the data loader (load_data) and every helper, so setup cells stay short.
"""
import os as _os

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

_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))


def set_seeds(seed=42):
    """Seed numpy and torch in one call. Returns the seed."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


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
__all__ = [
    'np', 'pd', 'torch', 'dl', 'COLORS',
    'set_seeds', 'load_data',
    'portfolio_metrics', 'build_metrics_table', 'running_dd',
    'plot_wealth', 'plot_epsilon_trajectory', 'plot_weight_heatmap',
    'plot_all_wealth', 'plot_drawdown', 'plot_summary_bars',
    'build_models', 'calibrate_models', 'fit_window', 'infer_window',
    'date_window', 'run_window', 'trained_params', 'model_report',
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
def plot_wealth(names, portfolios, title, figsize=(11, 5)):
    """Cumulative wealth curves for a list of (name, portfolio) pairs."""
    index = portfolios[0].rets.index
    # Mirror the library's wealth_plot: prepend a baseline wealth=1.0 anchor one
    # period before the first test date so curves start at 1.0 rather than 1+r0.
    anchor_date = (index[0] - pd.Timedelta(days=7)
                   if isinstance(index, pd.DatetimeIndex) else index[0] - 1)
    dates = index.insert(0, anchor_date)
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


def plot_weight_heatmap(po_rom, spo_rom, asset_labels, figsize=(16, 7)):
    """Side-by-side weight heatmaps for PO-ROM and SPO-ROM over time."""
    dates = spo_rom.portfolio.rets.index
    n_y   = len(asset_labels)
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    for ax, model, title, cmap in zip(
        axes,
        [po_rom, spo_rom],
        ['PO-ROM (fixed ε = 0.5)', 'SPO-ROM (learned ε)'],
        ['Blues', 'Reds']
    ):
        w         = model.portfolio.weights
        n_periods = w.shape[0]
        im        = ax.imshow(w.T, aspect='auto', cmap=cmap,
                              interpolation='nearest', vmin=0)
        tick_pos  = np.linspace(0, n_periods - 1, min(8, n_periods), dtype=int)
        tick_lab  = [dates[i].strftime('%Y-%m') for i in tick_pos]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab, rotation=45, fontsize=7)
        ax.set_yticks(range(n_y))
        ax.set_yticklabels(asset_labels, fontsize=7)
        ax.set_xlabel('Date')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='Weight')
    plt.suptitle('Portfolio Weights Over Time', fontsize=13)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Interactive Plotly plotters
# ---------------------------------------------------------------------------
def _dash(ls):
    return 'dash' if ls == '--' else 'solid'


def plot_all_wealth(all_names, all_ports):
    """Interactive cumulative wealth chart (Plotly). Legend placed outside right."""
    index = all_ports[0].rets.index
    # Mirror the library's wealth_plot: prepend a baseline wealth=1.0 anchor one
    # period before the first test date so curves start at 1.0 rather than 1+r0.
    anchor_date = (index[0] - pd.Timedelta(days=7)
                   if isinstance(index, pd.DatetimeIndex) else index[0] - 1)
    dates = [d.strftime('%Y-%m-%d') for d in index.insert(0, anchor_date)]
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

    cfg    : dict of hyperparameters (seed, epochs, lr, weight_decay, gamma_lr,
             max_weight, long_short, target_ratio, ...). Stashed as model.cfg.
    n_x    : number of features.
    n_y    : number of assets.
    n_obs  : rolling-window size.
    which  : optional iterable of model names to keep (default: all eight).
    returns: dict {name: model} with cfg attached to every model.
    """
    ew = bm.equal_weight(n_x=n_x, n_y=n_y, n_obs=n_obs)
    po_m = bm.pred_then_opt(n_x, n_y, n_obs, opt_layer='base_mod',
                            max_weight=cfg['max_weight'], long_short=cfg['long_short'],
                            set_seed=cfg['seed']).double()
    po_rom = bm.pred_then_opt(n_x, n_y, n_obs, opt_layer='base_rom', epsilon=0.5,
                              max_weight=cfg['max_weight'], long_short=cfg['long_short'],
                              set_seed=cfg['seed']).double()

    shared = dict(n_x=n_x, n_y=n_y, n_obs=n_obs, pred_model='linear',
                  epochs=cfg['epochs'], lr=cfg['lr'], weight_decay=cfg['weight_decay'],
                  max_weight=cfg['max_weight'], long_short=cfg['long_short'],
                  set_seed=cfg['seed'])
    models = {'EW': ew, 'PO-M': po_m, 'PO-ROM': po_rom}
    for name, spec in _SPO_SPECS:
        kw = dict(spec)
        if kw['opt_layer'] in ('nominal', 'tv', 'hellinger'):
            kw['gamma_lr'] = cfg['gamma_lr']
        models[name] = e2e_net(**shared, **kw).double()

    for m in models.values():
        m.cfg = cfg
    if which is not None:
        models = {k: models[k] for k in which}
    return models


def calibrate_models(models, X_train, Y_train, target_ratio=None):
    """One-time calibration of pred_loss_factor for every SPO (e2e_net) model.

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
            out[name] = m.calibrate_pred_loss_factor(X_train, Y_train, tr)
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


def _ols_init(model, X_train_df, Y_train_df):
    """Warm-start model.pred_layer with OLS weights (shared by fit_window).

    model       : pred_then_opt or e2e_net with a linear pred_layer.
    X_train_df  : feature DataFrame (no intercept column).
    Y_train_df  : return DataFrame, positionally aligned with X_train_df.
    returns     : Theta (n_y x [1+n_x]) OLS solution, [bias | weights].
    """
    Xt = X_train_df.copy()
    Xt.insert(0, 'ones', 1.0)
    device = model.pred_layer.weight.device
    X_cpu = torch.tensor(Xt.values, dtype=torch.double)
    Y_cpu = torch.tensor(Y_train_df.values, dtype=torch.double)
    Theta = torch.linalg.lstsq(X_cpu, Y_cpu).solution.T.to(device)
    with torch.no_grad():
        model.pred_layer.bias.copy_(Theta[:, 0])
        model.pred_layer.weight.copy_(Theta[:, 1:])
    return Theta


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
    reset       : if True, restore e2e_net learnable params from init_state_path
                  before training (ignored for the other model types).
    returns     : the (fitted) model, with introspection attributes attached.
    """
    if isinstance(model, bm.equal_weight):
        return model

    if isinstance(model, bm.pred_then_opt):
        Theta = _ols_init(model, X_train_df, Y_train_df)
        if model.model_type == 'base_rom':
            model.update_sigma_mu_hat(X_train_df)
        _stash_fit_attrs(model, X_train_df, Y_train_df, Theta)
        return model

    # e2e_net
    if reset and hasattr(model, 'init_state_path'):
        model.load_state_dict(torch.load(model.init_state_path))
    Theta = _ols_init(model, X_train_df, Y_train_df)
    if model.model_type == 'base_rom':
        model.update_sigma_mu_hat(X_train_df)
    train_set = DataLoader(pc.SlidingWindow(X_train_df, Y_train_df, model.n_obs, model.perf_period))
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
    X_df    : feature DataFrame covering [pred_start - n_obs ... pred_end (+1)].
    Y_df    : return DataFrame, positionally aligned with X_df.
    returns : pc.backtest with weights, rets, dates and computed stats().
    """
    n_obs = model.n_obs
    ds = pc.SlidingWindow(X_df, Y_df, n_obs, 1)
    is_ew = isinstance(model, bm.equal_weight) or not hasattr(model, 'forward')
    device = next(model.parameters()).device if not is_ew else None

    weights, rets, dates = [], [], []
    with torch.no_grad():
        for j in range(len(ds)):
            x, y, y_perf = ds[j]
            if is_ew:
                w = np.ones(model.n_y) / model.n_y
            else:
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
    lookback rows (and one trailing row) so infer_window can predict the full
    [pred_start, pred_end] range.

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
    b = min(p_end + 2, n)

    X_train_df = X.data.iloc[:t_end + 1]
    Y_train_df = Y.data.iloc[:t_end + 1]
    X_test_df = X.data.iloc[a:b]
    Y_test_df = Y.data.iloc[a:b]

    # Standardize on this window's train slice; apply to train + test (incl. the
    # n_obs lookback rows in X_test_df). The test stats come from the past only.
    X_train_df, X_test_df = _standardize(X_train_df, X_test_df)
    return X_train_df, Y_train_df, X_test_df, Y_test_df


def run_window(models, X, Y, train_end, pred_start=None, pred_end=None):
    """Fit every model on the train slice and infer on the holdout slice.

    models     : dict {name: model} from build_models (already calibrated).
    X, Y       : TrainTest objects from fetch_data_from_disk.
    train_end  : last training date (inclusive).
    pred_start : first prediction date (default: just after train_end).
    pred_end   : last prediction date (default: end of data).
    returns    : dict {name: pc.backtest} of out-of-sample portfolios.
    """
    Xtr, Ytr, Xte, Yte = date_window(X, Y, train_end, pred_start, pred_end)
    ports = {}
    for name, m in models.items():
        fit_window(m, Xtr, Ytr)
        ports[name] = infer_window(m, Xte, Yte)
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
