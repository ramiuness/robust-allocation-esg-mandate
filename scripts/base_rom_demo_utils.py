"""
Utilities for base_rom_demo.ipynb.
Metric computation, running drawdown, and all plot functions.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    dates = portfolios[0].rets.index
    fig, ax = plt.subplots(figsize=figsize)
    for name, port in zip(names, portfolios):
        color, ls = COLORS[name]
        ax.plot(dates, port.rets['tri'].values,
                color=color, linestyle=ls, linewidth=2, label=name)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Wealth')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()


def plot_drawdown(names, portfolios, title, figsize=(11, 4)):
    """Running drawdown curves for a list of (name, portfolio) pairs."""
    dates = portfolios[0].rets.index
    fig, ax = plt.subplots(figsize=figsize)
    for name, port in zip(names, portfolios):
        color, ls = COLORS[name]
        ax.plot(dates, running_dd(port).values,
                color=color, linestyle=ls, linewidth=1.5, label=name)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown')
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


def plot_summary_bars(all_names, all_metrics, figsize=(14, 5)):
    """Horizontal bar charts of Sharpe and Effective Holdings for all models."""
    bar_colors = [COLORS[n][0] for n in all_names]
    fig, axes  = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    ax.barh(all_names, all_metrics['Sharpe'].values, color=bar_colors, alpha=0.8)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Sharpe Ratio')
    ax.set_title('Sharpe Ratio')
    ax.grid(True, axis='x', alpha=0.3)

    ax = axes[1]
    ax.barh(all_names, all_metrics['Eff. Holdings'].values, color=bar_colors, alpha=0.8)
    ax.set_xlabel('Effective Holdings')
    ax.set_title('Effective Holdings')
    ax.grid(True, axis='x', alpha=0.3)

    plt.suptitle('Performance Summary', fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_all_wealth(all_names, all_ports, figsize=(12, 6)):
    """Cumulative wealth curves for all six models on a single axes."""
    dates = all_ports[0].rets.index
    fig, ax = plt.subplots(figsize=figsize)
    for name, port in zip(all_names, all_ports):
        color, ls = COLORS[name]
        lw = 2.5 if 'ROM' in name else 1.8
        ax.plot(dates, port.rets['tri'].values,
                color=color, linestyle=ls, linewidth=lw, label=name)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Wealth')
    ax.set_title('All Models: Cumulative Wealth')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()
