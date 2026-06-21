# API Documentation: e2edro Library

This document provides an overview of the `e2edro` (End-to-End Distributionally Robust Optimization) library used in the integrated learning validation study. **Note**: Source code files are excluded from the repository; this serves as a reference for methodology and reproducibility.

---

## Library Architecture

### Core Components

```
src/e2edro/
├── BaseModels.py       # Traditional optimization models
├── e2edro.py           # End-to-end learning models
├── DataLoad.py         # Data loading and generation
├── PortfolioClasses.py # Portfolio evaluation
├── PlotFunctions.py    # Visualization
├── LossFunctions.py    # Training objectives
└── RiskFunctions.py    # Risk measures
```

### Key Dependencies

| Package | Purpose |
|---------|---------|
| **torch** | Autograd, neural networks, optimization |
| **cvxpy** | Convex optimization modeling |
| **cvxpylayers** | Differentiable optimization layers |
| **pandas** | Data manipulation |
| **numpy** | Numerical computing |

---

## BaseModels.py

### Class: `equal_weight`

**Purpose**: Benchmark 1/n equally-weighted portfolio

```python
ew = equal_weight(n_x, n_y, n_obs)
ew.net_roll_test(X, Y, n_roll=4)
```

**Attributes**:
- `portfolio`: Portfolio object with performance metrics

---

### Class: `pred_then_opt`

**Purpose**: Traditional Predict-then-Optimize model with fixed risk aversion

```python
po = pred_then_opt(
    n_x,              # Number of features/factors
    n_y,              # Number of assets
    n_obs,            # Observation window size
    sigma,            # Covariance matrix (DataFrame or Tensor)
    kappa=1.0,        # Risk aversion coefficient (fixed)
    opt_layer='base_mv',  # Optimization layer type
    max_weight=None,  # Optional weight constraint
    set_seed=42       # Random seed
)
```

**Key Methods**:
- `forward(x)`: Forward pass through prediction and optimization
- `net_roll_test(X, Y, n_roll)`: Rolling window backtest

**Optimization Layer Options**:
- `'base_mv'`: Mean-Variance optimization

---

## e2edro.py

### Class: `e2e_net`

**Purpose**: End-to-end learning model with learnable parameters

```python
e2e = e2e_net(
    n_x,                    # Number of features
    n_y,                    # Number of assets
    n_obs,                  # Observation window
    sigma,                  # Covariance matrix (Tensor)
    opt_layer='base_mv',    # Optimization layer
    pred_model='linear',    # Prediction model type
    pred_loss_factor=0.25,  # Weight for prediction loss
    epochs=10,              # Training epochs per window
    lr=1e-3,                # Learning rate (for prediction layer)
    kappa_lr=0.1,           # Separate LR for kappa (recommended for meaningful learning)
    train_kappa=True,       # Learn risk aversion
    max_weight=None,        # Weight constraint
    set_seed=42             # Random seed
)
```

**Note on Kappa Learning**: Use `kappa_lr=0.1` for meaningful kappa learning (~23% change in 10 epochs vs 0.3% with default lr=1e-3). This separate learning rate addresses the slow convergence of kappa due to small gradients from the cvxpylayers optimization layer.

**Learnable Parameters**:
- `kappa`: Risk aversion coefficient (when `train_kappa=True`)
- Linear layer weights for return prediction

**Key Methods**:
- `forward(x)`: Forward pass with differentiable optimization
- `net_roll_test(X, Y, n_roll)`: Rolling backtest with training

**Training Loop** (per rolling window):
```python
for epoch in range(epochs):
    optimizer.zero_grad()
    w = model(x_train)
    loss = decision_loss(w, y_train) + pred_loss_factor * pred_loss
    loss.backward()
    optimizer.step()
```

---

## DataLoad.py

### Function: `fetch_market_data`

**Purpose**: Download real market data from Yahoo Finance and Fama-French

```python
X, Y = fetch_market_data(
    start='2000-07-01',
    end='2025-01-01',
    split=[0.6, 0.4]  # Train/test split
)
```

**Returns**:
- `X`: FactorData object (Fama-French factors)
- `Y`: ReturnData object (stock returns)

**Asset Universe** (20 stocks):
- Technology: AAPL, MSFT, AMZN
- Financials: C, JPM, BAC
- Energy: XOM, HAL
- Consumer: MCD, WMT, COST
- Industrials: CAT, LMT
- Healthcare: JNJ, PFE
- Communication: DIS, VZ, T
- Utilities: ED
- Materials: NEM

**Factors** (8 Fama-French):
- Market, SMB, HML, RMW, CMA
- Momentum (MOM)
- Short-term Reversal (ST_Rev)
- Long-term Reversal (LT_Rev)

---

### Function: `synthetic_market_calibrated`

**Purpose**: Generate calibrated synthetic data with known properties

```python
X, Y = synthetic_market_calibrated(
    n_x=8,                    # Number of factors
    n_y=20,                   # Number of assets
    n_tot=665,                # Total observations
    n_obs=52,                 # Window size
    split=[0.6, 0.4],         # Train/test split
    set_seed=42,              # Random seed
    target_mean=None,         # Target return mean
    target_cov=None,          # Target covariance
    vol_regime_changes=True   # Enable regime switching
)
```

**Return Generation Model**:
```
Y_t = α + X_t·β + ε_gaussian + ε_shock + ε_sector
```

Where:
- α: Asset-specific expected returns (0.05%-0.30% weekly)
- X_t·β: Factor exposures
- ε_gaussian: Gaussian idiosyncratic noise
- ε_shock: Fat-tailed exponential shocks
- ε_sector: Sector correlation component

**Volatility Regimes**:
- Regime 1 (low vol): multiplier = 0.8×
- Regime 2 (normal): multiplier = 1.0×
- Regime 3 (high vol): multiplier = 1.2×

---

## PortfolioClasses.py

### Class: `Portfolio`

**Purpose**: Store and compute portfolio performance metrics

**Metrics Computed**:
```python
portfolio.sharpe           # Sharpe ratio
portfolio.sortino          # Sortino ratio
portfolio.annualized_return  # Annualized return
portfolio.vol              # Volatility
portfolio.max_drawdown     # Maximum drawdown
portfolio.turnover         # Portfolio turnover
portfolio.effective_holdings  # Diversification measure
```

**Key Methods**:
- `compute_metrics()`: Calculate all performance metrics
- `rets`: DataFrame with returns and cumulative wealth (TRI)

---

### Class: `ReturnData` / `FactorData`

**Purpose**: Data containers with train/test split functionality

```python
Y.train()  # Training data (DataFrame)
Y.test()   # Test data (DataFrame)
Y.data     # Full dataset
Y.n_obs    # Observation window
```

---

## LossFunctions.py

### Decision Loss

**Purpose**: Portfolio-aware loss for end-to-end training

```python
# Negative portfolio return (to minimize)
decision_loss = -torch.mean(torch.sum(w * y, dim=1))
```

### Prediction Loss

**Purpose**: Standard MSE for return prediction

```python
pred_loss = torch.mean((y_pred - y_actual)**2)
```

### Combined Loss

```python
total_loss = decision_loss + pred_loss_factor * pred_loss
```

The `pred_loss_factor` (default 0.25) balances decision quality with prediction accuracy.

---

## RiskFunctions.py

### Function: `compute_cvar`

**Purpose**: Compute Conditional Value-at-Risk

```python
cvar = compute_cvar(returns, alpha=0.05)
```

---

## Optimization Layer Details

### Mean-Variance (base_mv)

**Formulation**:
```
maximize:  μᵀw - (κ/2) wᵀΣw
subject to: w ≥ 0
            1ᵀw = 1
            w ≤ max_weight (optional)
```

**Implementation**: cvxpy + cvxpylayers for differentiability

```python
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

w = cp.Variable(n_y)
mu = cp.Parameter(n_y)
kappa = cp.Parameter(1, nonneg=True)

objective = cp.Maximize(mu @ w - (kappa/2) * cp.quad_form(w, Sigma))
constraints = [w >= 0, cp.sum(w) == 1]
if max_weight:
    constraints.append(w <= max_weight)

problem = cp.Problem(objective, constraints)
layer = CvxpyLayer(problem, parameters=[mu, kappa], variables=[w])
```

### Differentiable Backward Pass

cvxpylayers computes gradients through the optimization using implicit differentiation, enabling end-to-end learning of:
- Prediction model parameters
- Risk aversion coefficient κ

---

## Usage Example

```python
import e2edro.DataLoad as dl
from e2edro.e2edro import e2e_net
from e2edro.BaseModels import pred_then_opt, equal_weight
import torch

# Load data
X, Y = dl.fetch_market_data(start='2000-07-01', split=[0.6, 0.4])
n_x, n_y, n_obs = X.train().shape[1], Y.train().shape[1], X.n_obs

# Initialize models
sigma = torch.tensor(Y.train().cov().values, dtype=torch.double)

ew = equal_weight(n_x, n_y, n_obs)
po = pred_then_opt(n_x, n_y, n_obs, sigma=Y.train().cov(), kappa=1.0)
e2e = e2e_net(n_x, n_y, n_obs, sigma=sigma, train_kappa=True, kappa_lr=0.1)

# Run backtests
ew.net_roll_test(X, Y, n_roll=4)
po.net_roll_test(X, Y, n_roll=4)
e2e.net_roll_test(X, Y, n_roll=4)

# Compare results
print(f"EW Sharpe: {ew.portfolio.sharpe:.4f}")
print(f"PO Sharpe: {po.portfolio.sharpe:.4f}")
print(f"E2E Sharpe: {e2e.portfolio.sharpe:.4f}")
print(f"Learned kappa: {e2e.kappa.item():.4f}")
```

---

## Original Source

This library is adapted from the **E2E-DRO** repository by Iyengar Lab (Columbia University):

- **Repository**: [github.com/Iyengar-Lab/E2E-DRO](https://github.com/Iyengar-Lab/E2E-DRO)
- **Paper**: [Distributionally Robust End-to-End Portfolio Construction](https://arxiv.org/abs/2206.05134)
- **Authors**: Giorgio Costa, Garud Iyengar
- **Published**: Quantitative Finance, Vol. 23, No. 10 (2023)

---

## Reference Papers

The implementation is based on the following research:

1. **Distributionally Robust End-to-End Portfolio Construction** (Costa & Iyengar, 2023): Core framework for robust E2E learning
2. **Decision-Focused Learning**: End-to-end optimization through the decision layer
3. **OptNet**: Differentiable optimization as a neural network layer
4. **cvxpylayers**: Differentiable convex optimization layers
5. **SPO (Smart Predict then Optimize)**: Task-loss minimization framework

See `references/` directory for full papers.

---

**Documentation Generated**: January 2025
**Library**: e2edro (End-to-End Distributionally Robust Optimization)
