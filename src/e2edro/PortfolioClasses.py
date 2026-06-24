# PortfolioClasses Module
#
####################################################################################################
## Import libraries
####################################################################################################
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable

####################################################################################################
# SlidingWindow torch Dataset to index data to use a sliding window
####################################################################################################
class SlidingWindow(Dataset):
    """Sliding window dataset constructor
    """
    def __init__(self, X, Y, n_obs, perf_period):
        """Construct a sliding (i.e., rolling) window dataset from a complete timeseries dataset

        Inputs
        X: pandas dataframe with the complete feature dataset
        Y: pandas dataframe with the complete asset return dataset
        n_obs: Number of scenarios in the window
        perf_period: Number of scenarios in the 'performance window' used to evaluate out-of-sample
        performance. The 'performance window' is also a sliding window

        Output
        Dataset where each element is the tuple (x, y, y_perf)
        x: Feature window (dim: [n_obs+1] x n_x)
        y: Realizations window (dim: n_obs x n_y)
        y_perf: Window of forward-looking (i.e., future) realizations (dim: perf_period x n_y)

        Note: For each feature window 'x', the last scenario x_t is reserved for prediction and
        optimization. Therefore, no pair in 'y' is required (it is assumed the pair y_T is not yet
        observable)
        """
        self.X = Variable(torch.tensor(X.values, dtype=torch.double))
        self.Y = Variable(torch.tensor(Y.values, dtype=torch.double))
        self.n_obs = n_obs
        self.perf_period = perf_period

    def __getitem__(self, index):
        x = self.X[index:index+self.n_obs+1]
        y = self.Y[index:index+self.n_obs]
        y_perf = self.Y[index+self.n_obs : index+self.n_obs+self.perf_period]
        return (x, y, y_perf)

    def __len__(self):
        return len(self.X) - self.n_obs - self.perf_period

####################################################################################################
# Backtest object to store out-of-sample results
####################################################################################################
class backtest:
    """backtest object
    """
    def __init__(self, len_test, n_y, dates):
        """Portfolio object. Stores the NN out-of-sample results

        Inputs
        len_test: Number of scenarios in the out-of-sample evaluation period
        n_y: Number of assets in the portfolio
        dates: DatetimeIndex

        Output
        Backtest object with fields:
        weights: Asset weights per period (dim: len_test x n_y)
        rets: Realized portfolio returns (dim: len_test x 1)
        tri: Total return index (i.e., absolute cumulative return) (dim: len_test x 1)
        mean: Average return over the out-of-sample evaluation period (dim: scalar)
        vol: Volatility (i.e., standard deviation of the returns) (dim: scalar)
        sharpe: pseudo-Sharpe ratio defined as 'mean / vol' (dim: scalar)
        sortino: Sortino ratio (return / downside deviation) (dim: scalar)
        max_drawdown: Maximum peak-to-trough decline (dim: scalar)
        turnover: Average portfolio turnover (dim: scalar)
        effective_holdings: Average effective number of holdings (dim: scalar)
        annualized_return: Compound annual growth rate (CAGR) (dim: scalar)
        """
        self.weights = np.zeros((len_test, n_y))
        self.rets = np.zeros(len_test)
        self.dates = dates[-len_test:]

    def stats(self):
        """Compute and store portfolio statistics.

        Computes: mean, vol, sharpe, sortino, max_drawdown, turnover, effective_holdings
        Then converts self.rets to DataFrame with columns ['rets', 'tri'].
        """
        if isinstance(self.rets, pd.DataFrame):
            return

        # Compute all metrics while self.rets is still a numpy array
        tri = np.cumprod(self.rets + 1)

        # Mean and volatility
        self.mean = (tri[-1])**(1/len(tri)) - 1
        self.vol = np.std(self.rets)

        # Sharpe ratio
        self.sharpe = self.mean / self.vol if self.vol > 0 else 0.0

        # Sortino ratio (return / downside deviation)
        downside_rets = self.rets[self.rets < 0]
        downside_std = np.std(downside_rets) if len(downside_rets) > 0 else 0.0
        self.sortino = self.mean / downside_std if downside_std > 0 else np.inf

        # Max drawdown
        running_max = np.maximum.accumulate(tri)
        drawdown = (tri - running_max) / running_max
        self.max_drawdown = float(drawdown.min())

        # Turnover
        if len(self.weights) < 2:
            self.turnover = 0.0
        else:
            turnovers = [np.sum(np.abs(self.weights[i] - self.weights[i-1]))
                        for i in range(1, len(self.weights))]
            self.turnover = float(np.mean(turnovers))

        # Effective holdings (inverse Herfindahl index)
        eff = [1.0 / np.sum(w**2) for w in self.weights if np.sum(w**2) > 0]
        self.effective_holdings = float(np.mean(eff)) if eff else 0.0

        # Annualized return (CAGR) - assuming weekly data (52 periods/year)
        n_periods = len(tri)
        periods_per_year = 52  # weekly data
        final_wealth = tri[-1]
        self.annualized_return = float(final_wealth ** (periods_per_year / n_periods) - 1)

        # Convert rets to DataFrame (must be last - changes self.rets type)
        self.rets = pd.DataFrame({'Date': self.dates, 'rets': self.rets, 'tri': tri})
        self.rets = self.rets.set_index('Date')

    def information_ratio(self, benchmark_returns):
        """Information ratio (excess return / tracking error).

        Parameters
        ----------
        benchmark_returns : array-like
            Benchmark returns to compare against.

        Returns
        -------
        float
            Information ratio
        """
        returns = self.rets['rets'].values
        benchmark_returns = np.asarray(benchmark_returns)
        if len(benchmark_returns) != len(returns):
            raise ValueError(f"Benchmark length ({len(benchmark_returns)}) must match "
                           f"returns length ({len(returns)})")
        excess_returns = returns - benchmark_returns
        tracking_error = np.std(excess_returns)
        if tracking_error == 0:
            return np.inf
        return float(np.mean(excess_returns) / tracking_error)

    def summary(self, name=None):
        """Print a summary of portfolio performance metrics.

        Parameters
        ----------
        name : str, optional
            Portfolio name for display

        Returns
        -------
        pd.Series
            Series containing all metrics
        """
        metrics = {
            'Mean Return': self.mean,
            'Volatility': self.vol,
            'Sharpe Ratio': self.sharpe,
            'Sortino Ratio': self.sortino,
            'Max Drawdown': self.max_drawdown,
            'Turnover': self.turnover,
            'Eff. Holdings': self.effective_holdings,
            'Annualized Return': self.annualized_return
        }
        series = pd.Series(metrics, name=name)

        header = f"Portfolio: {name}" if name else "Portfolio Summary"
        print(f"\n{'='*50}")
        print(header)
        print('='*50)
        print(f"  Mean Return:       {self.mean:>12.4f}")
        print(f"  Volatility:        {self.vol:>12.4f}")
        print(f"  Sharpe Ratio:      {self.sharpe:>12.4f}")
        print(f"  Sortino Ratio:     {self.sortino:>12.4f}")
        print(f"  Max Drawdown:      {self.max_drawdown:>12.4f}")
        print(f"  Turnover:          {self.turnover:>12.4f}")
        print(f"  Eff. Holdings:     {self.effective_holdings:>12.2f}")
        print(f"  Annualized Return: {self.annualized_return:>12.2%}")
        print('='*50)

        return series

####################################################################################################
# InSample object to store in-sample results
####################################################################################################
class InSample:
    """InSample object
    """
    def __init__(self):
        """Portfolio object. Stores the NN in-sample results

        Output
        InSample object with fields:
        loss: Empty list to hold the training loss after each forward pass
        gamma: Empty list to hold the gamma value after each backward pass
        delta: Empty list to hold the delta value after each backward pass
        val_loss (optional): Empty list to hold the valildation loss after each forward pass
        """
        self.loss = []
        self.gamma = []
        self.delta = []
        self.val_loss = []

    def df(self):
        """Return a pandas dataframe object by merging the self.lists
        """
        if not self.delta and not self.val_loss:
            return pd.DataFrame(list(zip(self.loss, self.gamma)), columns=['loss', 'gamma'])
        elif not self.delta:
            return pd.DataFrame(list(zip(self.loss, self.val_loss, self.gamma)), 
                            columns=['loss', 'val_loss', 'gamma'])
        elif not self.val_loss:
            return pd.DataFrame(list(zip(self.loss, self.gamma, self.delta)), 
                            columns=['loss', 'gamma', 'delta'])
        else:
            return pd.DataFrame(list(zip(self.loss, self.val_loss, self.gamma, self.delta)), 
                            columns=['loss', 'val_loss', 'gamma', 'delta'])


####################################################################################################
# Backtest object to store out-of-sample results
####################################################################################################
class CrossVal:
    """Portfolio object
    """
    def __init__(self):
        """CrossVal object. Stores the NN in-sample cross validation results

        Output
        CrossVal object with fields:
        lr: Empty list to hold the learning rate of this run
        epochs: Empty list to hold the number of epochs in this run
        train_loss: Empty list to hold the average training loss of all folds
        val_loss: Empty list to hold the average validation loss of all folds
        """
        self.lr = []
        self.epochs = []
        self.val_loss = []

    def df(self):
        """Return a pandas dataframe object by merging the self.lists
        """
        return pd.DataFrame(list(zip(self.lr, self.epochs, self.val_loss)), 
                            columns=['lr', 'epochs', 'val_loss'])

