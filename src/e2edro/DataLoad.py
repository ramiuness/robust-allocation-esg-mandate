# DataLoad module
#
####################################################################################################
## Import libraries
####################################################################################################
import torch
import torch.nn as nn
import pandas as pd
import pandas_datareader as pdr
import numpy as np
import time
import statsmodels.api as sm

import yfinance as yf

####################################################################################################
# TrainTest class
####################################################################################################
class TrainTest:
    def __init__(self, data, n_obs, split):
        """Object to hold the training, validation and testing datasets

        Inputs
        data: pandas dataframe with time series data
        n_obs: Number of observations per batch
        split: list of ratios that control the partition of data into training, testing and 
        validation sets. 
    
        Output. TrainTest object with fields and functions:
        data: Field. Holds the original pandas dataframe
        train(): Function. Returns a pandas dataframe with the training subset of observations
        """
        self.data = data
        self.n_obs = n_obs
        self.split = split

        n_obs_tot = self.data.shape[0]
        numel = n_obs_tot * np.cumsum(split)
        self.numel = [round(i) for i in numel]

    def split_update(self, split):
        """Update the list outlining the split ratio of training, validation and testing
        """
        self.split = split
        n_obs_tot = self.data.shape[0]
        numel = n_obs_tot * np.cumsum(split)
        self.numel = [round(i) for i in numel]

    def train(self):
        """Return the training subset of observations
        """
        return self.data[:self.numel[0]]

    def test(self):
        """Return the test subset of observations
        """
        return self.data[self.numel[0]-self.n_obs:self.numel[1]]

####################################################################################################
# Generate linear synthetic data
####################################################################################################
def synthetic(n_x=5, n_y=10, n_tot=1200, n_obs=104, split=[0.6, 0.4], set_seed=100):
    """Generates synthetic (normally-distributed) asset and factor data

    Inputs
    n_x: Integer. Number of features
    n_y: Integer. Number of assets
    n_tot: Integer. Number of observations in the whole dataset
    n_obs: Integer. Number of observations per batch
    split: List of floats. Train-validation-test split as percentages (must sum up to one)
    set_seed: Integer. Used for replicability of the numpy RNG.

    Outputs
    X: TrainValTest object with feature data split into train, validation and test subsets
    Y: TrainValTest object with asset data split into train, validation and test subsets
    """
    np.random.seed(set_seed)

    # 'True' prediction bias and weights
    a = np.sort(np.random.rand(n_y) / 250) + 0.0001
    b = np.random.randn(n_x, n_y) / 5
    c = np.random.randn(int((n_x+1)/2), n_y)

    # Noise std dev
    s = np.sort(np.random.rand(n_y))/20 + 0.02

    # Syntehtic features
    X = np.random.randn(n_tot, n_x) / 50
    X2 = np.random.randn(n_tot, int((n_x+1)/2)) / 50

    # Synthetic outputs
    Y = a + X @ b + X2 @ c + s * np.random.randn(n_tot, n_y)

    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)
    
    # Partition dataset into training and testing sets
    return TrainTest(X, n_obs, split), TrainTest(Y, n_obs, split)

####################################################################################################
# Generate non-linear synthetic data
####################################################################################################
def synthetic_nl(n_x=5, n_y=10, n_tot=1200, n_obs=104, split=[0.6, 0.4], set_seed=100):
    """Generates synthetic (normally-distributed) factor data and mix them following a quadratic 
    model of linear, squared and cross products to produce the asset data. 

    Inputs
    n_x: Integer. Number of features
    n_y: Integer. Number of assets
    n_tot: Integer. Number of observations in the whole dataset
    n_obs: Integer. Number of observations per batch
    split: List of floats. Train-validation-test split as percentages (must sum up to one)
    set_seed: Integer. Used for replicability of the numpy RNG.

    Outputs
    X: TrainValTest object with feature data split into train, validation and test subsets
    Y: TrainValTest object with asset data split into train, validation and test subsets
    """
    np.random.seed(set_seed)

    # 'True' prediction bias and weights
    a = np.sort(np.random.rand(n_y) / 200) + 0.0005
    b = np.random.randn(n_x, n_y) / 4
    c = np.random.randn(int((n_x+1)/2), n_y)
    d = np.random.randn(n_x**2, n_y) / n_x

    # Noise std dev
    s = np.sort(np.random.rand(n_y))/20 + 0.02

    # Syntehtic features
    X = np.random.randn(n_tot, n_x) / 50
    X2 = np.random.randn(n_tot, int((n_x+1)/2)) / 50
    X_cross = 100 * (X[:,:,None] * X[:,None,:]).reshape(n_tot, n_x**2)
    X_cross = X_cross - X_cross.mean(axis=0)

    # Synthetic outputs
    Y = a + X @ b + X2 @ c + X_cross @ d + s * np.random.randn(n_tot, n_y)

    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)
    
    # Partition dataset into training and testing sets
    return TrainTest(X, n_obs, split), TrainTest(Y, n_obs, split)

####################################################################################################
# Generate non-linear synthetic data
####################################################################################################
def synthetic_NN(n_x=5, n_y=10, n_tot=1200, n_obs=104, split=[0.6, 0.4], set_seed=45678):
    """Generates synthetic (normally-distributed) factor data and mix them following a 
    randomly-initialized 3-layer neural network. 

    Inputs
    n_x: Integer. Number of features
    n_y: Integer. Number of assets
    n_tot: Integer. Number of observations in the whole dataset
    n_obs: Integer. Number of observations per batch
    split: List of floats. Train-validation-test split as percentages (must sum up to one)
    set_seed: Integer. Used for replicability of the numpy RNG.

    Outputs
    X: TrainValTest object with feature data split into train, validation and test subsets
    Y: TrainValTest object with asset data split into train, validation and test subsets
    """
    np.random.seed(set_seed)

    # Syntehtic features
    X = np.random.randn(n_tot, n_x) * 10 + 0.5
    
    # Initialize NN object
    synth = synthetic3layer(n_x, n_y, set_seed).double()

    # Synthetic outputs
    Y = synth(torch.from_numpy(X))

    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y.detach().numpy()) / 10
    
    # Partition dataset into training and testing sets
    return TrainTest(X, n_obs, split), TrainTest(Y, n_obs, split)

####################################################################################################
# E2E neural network module
####################################################################################################
class synthetic3layer(nn.Module):
    """End-to-end DRO learning neural net module.
    """
    def __init__(self, n_x, n_y, set_seed):
        """End-to-end learning neural net module

        This NN module implements a linear prediction layer 'pred_layer' and a DRO layer 
        'opt_layer' based on a tractable convex formulation from Ben-Tal et al. (2013). 'delta' and
        'gamma' are declared as nn.Parameters so that they can be 'learned'.

        Inputs
        n_x: Number of inputs (i.e., features) in the prediction model
        n_y: Number of outputs from the prediction model

        Output
        e2e_net: nn.Module object 
        """
        super(synthetic3layer, self).__init__()

        # Set random seed (to be used for replicability of numerical experiments)
        torch.manual_seed(set_seed)

        # Neural net with 3 hidden layers 
        self.pred_layer = nn.Sequential(nn.Linear(n_x, int(0.5*(n_x+n_y))),
                    nn.ReLU(),
                    nn.Linear(int(0.5*(n_x+n_y)), int(0.6*(n_x+n_y))),
                    nn.ReLU(),
                    nn.Linear(int(0.6*(n_x+n_y)), n_y),
                    nn.ReLU(),
                    nn.Linear(n_y, n_y))

    #-----------------------------------------------------------------------------------------------
    # forward: forward pass of the synthetic3layer NN
    #-----------------------------------------------------------------------------------------------
    def forward(self, X):
        """Forward pass of the NN module

        Inputs
        X: Features. (n_obs x n_x) torch tensor with feature timeseries data

        Outputs
        Y: Syntheticly generated output. (n_obs x n_y) torch tensor of outputs
        """
        Y = torch.stack([self.pred_layer(x_t) for x_t in X])

        return Y

####################################################################################################
# Synthetic data with Gaussian and exponential noise terms
####################################################################################################
def synthetic_exp(n_x=5, n_y=10, n_tot=1200, n_obs=104, split=[0.6, 0.4], set_seed=123):

    np.random.seed(set_seed)
    
    # Exponential (shock) noise term
    exp_noise = 0.2 * np.random.choice([-1,0,1], p=[0.15, 0.7, 0.15], 
                                        size=(n_tot, n_y)) * np.random.exponential(1,(n_tot, n_y))
    exp_noise = exp_noise.clip(-0.3, 0.3)

    # Gaussian noise term
    gauss_noise = 0.2 * np.random.randn(n_tot, n_y)

    # 'True' prediction bias and weights
    alpha = np.sort(np.random.rand(n_y).clip(0.2,1) / 1000)
    beta = np.random.randn(n_x, n_y).clip(-3,3) / n_x

    # Syntehtic features
    X = np.random.randn(n_tot, n_x).clip(-3,3) / 10

    # Synthetic outputs
    Y = (alpha + X @ beta + exp_noise + gauss_noise).clip(-0.2,0.3) / 15

    # Convert to dataframes
    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)
    
    # Partition dataset into training and testing sets
    return TrainTest(X, n_obs, split), TrainTest(Y, n_obs, split)

####################################################################################################
# Synthetic data calibrated to match market statistics
####################################################################################################
def synthetic_market_calibrated(n_x=8, n_y=20, n_tot=665, n_obs=104, split=[0.6, 0.4],
                                set_seed=123, target_mean=None, target_cov=None,
                                vol_regime_changes=True):
    """Generate synthetic market data calibrated to match target statistics

    This function generates realistic synthetic market data with:
    - Multi-factor structure mimicking Fama-French factors
    - Fat-tailed returns (exponential shocks)
    - Heterogeneous asset characteristics
    - Optional volatility regime changes (for testing covariance updates)

    The data can be calibrated to match exact target mean and covariance, or use
    default market-like parameters.

    Parameters
    ----------
    n_x : int
        Number of factors (features)
    n_y : int
        Number of assets
    n_tot : int
        Total number of observations
    n_obs : int
        Sliding window size
    split : list
        Train/test split ratios [train, test]
    set_seed : int
        Random seed for reproducibility
    target_mean : np.ndarray or None
        Target mean returns (n_y,). If None, uses default ~0.28% weekly
    target_cov : np.ndarray or None
        Target covariance matrix (n_y, n_y). If None, uses default market-like structure
    vol_regime_changes : bool
        If True, adds volatility regime shifts to make covariance time-varying

    Returns
    -------
    X : TrainTest
        Factor data (features)
    Y : TrainTest
        Return data (targets)

    Examples
    --------
    >>> # Use default market-like statistics
    >>> X, Y = synthetic_market_calibrated(n_x=8, n_y=20, set_seed=42)

    >>> # Calibrate to specific target statistics
    >>> X_real, Y_real = dl.fetch_market_data(...)
    >>> target_mean = Y_real.train().mean().values
    >>> target_cov = Y_real.train().cov().values
    >>> X_synth, Y_synth = synthetic_market_calibrated(
    ...     target_mean=target_mean, target_cov=target_cov
    ... )
    """
    np.random.seed(set_seed)

    # ============================================================================
    # STAGE 1: Generate base synthetic data with realistic factor structure
    # ============================================================================

    # Factor generation: standardized with small scale
    X = np.random.randn(n_tot, n_x) / 10

    # Heterogeneous factor loadings (beta diversity across assets)
    # Market factor (first) has strongest influence
    beta = np.random.randn(n_x, n_y) / 5
    beta[0, :] *= 2.0  # Market factor has 2x influence

    # Add asset heterogeneity: different sensitivity ranges
    asset_sensitivity = np.linspace(0.6, 1.4, n_y)
    beta *= asset_sensitivity

    # Asset-specific means (alpha)
    alpha = np.sort(np.random.uniform(0.0005, 0.003, n_y))

    # ============================================================================
    # STAGE 2: Generate multi-component noise
    # ============================================================================

    # Component 1: Gaussian idiosyncratic noise
    gaussian_noise = 0.015 * np.random.randn(n_tot, n_y)

    # Component 2: Exponential shocks (fat tails) - similar to synthetic_exp
    exp_shocks = 0.15 * np.random.choice(
        [-1, 0, 1],
        p=[0.15, 0.7, 0.15],
        size=(n_tot, n_y)
    ) * np.random.exponential(1, (n_tot, n_y))
    exp_shocks = exp_shocks.clip(-0.25, 0.25)

    # Component 3: Sector correlations (create 4 sectors)
    n_sectors = 4
    sector_shocks = np.zeros((n_tot, n_y))
    assets_per_sector = n_y // n_sectors
    sector_noise = 0.01 * np.random.randn(n_tot, n_sectors)

    for i in range(n_sectors):
        start_idx = i * assets_per_sector
        end_idx = start_idx + assets_per_sector if i < n_sectors - 1 else n_y
        sector_shocks[:, start_idx:end_idx] = sector_noise[:, i:i+1]

    # ============================================================================
    # STAGE 3: Optional volatility regime changes
    # ============================================================================

    if vol_regime_changes:
        # Create 3 regimes with different volatility multipliers
        regime_length = n_tot // 3
        vol_multiplier = np.ones(n_tot)
        vol_multiplier[:regime_length] = 0.8          # Low vol regime
        vol_multiplier[regime_length:2*regime_length] = 1.0  # Normal vol
        vol_multiplier[2*regime_length:] = 1.2        # High vol regime

        # Apply to noise components
        gaussian_noise *= vol_multiplier[:, np.newaxis]
        exp_shocks *= vol_multiplier[:, np.newaxis]

    # ============================================================================
    # STAGE 4: Combine components to form base returns
    # ============================================================================

    Y_base = alpha + X @ beta + gaussian_noise + exp_shocks + sector_shocks

    # ============================================================================
    # STAGE 5: Calibration to target statistics
    # ============================================================================

    if target_mean is not None and target_cov is not None:
        # Exact calibration via affine transformation
        Y_calibrated = _calibrate_to_target(Y_base, target_mean, target_cov)
    else:
        # Use default market-like scaling
        # Target: ~0.28% weekly mean, ~2.1% weekly vol
        default_mean = 0.0028
        default_vol = 0.021

        # Simple scaling to match typical market statistics
        current_mean = Y_base.mean(axis=0).mean()
        current_vol = Y_base.std(axis=0).mean()

        scale = default_vol / current_vol
        shift = default_mean - (current_mean * scale)

        Y_calibrated = Y_base * scale + shift

    # ============================================================================
    # STAGE 6: Convert to DataFrames and return TrainTest objects
    # ============================================================================

    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y_calibrated)

    return TrainTest(X, n_obs, split), TrainTest(Y, n_obs, split)


def _calibrate_to_target(Y_base, target_mean, target_cov):
    """Calibrate data to exact target mean and covariance via affine transformation

    Uses Cholesky decomposition to find transformation matrix L such that:
        Y_calibrated = L @ (Y_base - mean_base) + target_mean
        cov(Y_calibrated) = target_cov

    Parameters
    ----------
    Y_base : np.ndarray
        Base synthetic data (n_obs, n_assets)
    target_mean : np.ndarray
        Target mean vector (n_assets,)
    target_cov : np.ndarray
        Target covariance matrix (n_assets, n_assets)

    Returns
    -------
    Y_calibrated : np.ndarray
        Calibrated data with exact target statistics
    """
    # Current statistics
    mean_base = Y_base.mean(axis=0)
    cov_base = np.cov(Y_base.T)

    # Add small regularization to ensure positive definite
    cov_base += 1e-8 * np.eye(cov_base.shape[0])
    target_cov_reg = target_cov + 1e-8 * np.eye(target_cov.shape[0])

    # Compute transformation matrix via Cholesky decomposition
    try:
        L_base = np.linalg.cholesky(cov_base)
        L_target = np.linalg.cholesky(target_cov_reg)
        L = L_target @ np.linalg.inv(L_base)
    except np.linalg.LinAlgError:
        # Fallback: use simple scaling if Cholesky fails
        print("Warning: Cholesky decomposition failed, using simple scaling")
        vol_base = np.sqrt(np.diag(cov_base))
        vol_target = np.sqrt(np.diag(target_cov))
        scale = vol_target / (vol_base + 1e-8)
        Y_centered = Y_base - mean_base
        Y_calibrated = Y_centered * scale + target_mean
        return Y_calibrated

    # Apply affine transformation
    Y_centered = Y_base - mean_base
    Y_calibrated = (L @ Y_centered.T).T + target_mean

    return Y_calibrated

####################################################################################################
# Fetch market data from Kenneth French's data library and Yahoo Finance
# https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
####################################################################################################
def fetch_market_data(start:str, end:str, split:list, freq:str='weekly', n_obs:int=104, n_y=None,
                      use_cache:bool=False, save_results:bool=False):
    """Fetch historical market data from Kenneth French's data library and Yahoo Finance.

    Downloads Fama-French factors as features and stock returns from Yahoo Finance.
    https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

    Parameters
    ----------
    start : str
        Start date of time series.
    end : str
        End date of time series.
    split : list
        Train-validation-test split as percentages.
    freq : str, optional
        Data frequency (daily, weekly, monthly). The default is 'weekly'.
    n_obs : int, optional
        Number of observations per batch. The default is 104.
    n_y : int, optional
        Number of assets to select. If None, all 20 assets are used. The default is None.
    use_cache : bool, optional
        State whether to load cached data or download data. The default is False.
    save_results : bool, optional
        State whether the data should be cached for future use. The default is False.

    Returns
    -------
    X: TrainTest
        TrainTest object with feature data split into train, validation and test subsets.
    Y: TrainTest
        TrainTest object with asset data split into train, validation and test subsets.
    """

    if use_cache:
        X = pd.read_pickle('./cache/factor_'+freq+'.pkl')
        Y = pd.read_pickle('./cache/asset_'+freq+'.pkl')
    else:
        tick_list = ['AAPL', 'MSFT', 'AMZN', 'C', 'JPM', 'BAC', 'XOM', 'HAL', 'MCD', 'WMT', 'COST',
                     'CAT', 'LMT', 'JNJ', 'PFE', 'DIS', 'VZ', 'T', 'ED', 'NEM']

        if n_y is not None:
            tick_list = tick_list[:n_y]

        # Download asset data (increased timeout for large historical downloads)
        data = yf.download(tick_list, start='1999-1-1', end=end, ignore_tz=True, timeout=60)
        Y = data['Close']        
        Y = Y.pct_change().dropna(axis=0)
        Y = Y[start:end]
        #Y.columns = tick_list

        # Download factor data 
        dl_freq = '_daily'
        X = pdr.get_data_famafrench('F-F_Research_Data_5_Factors_2x3'+dl_freq, start=start,
                    end=end)[0]
        rf_df = X['RF']
        X = X.drop(['RF'], axis=1)
        mom_df = pdr.get_data_famafrench('F-F_Momentum_Factor'+dl_freq, start=start, end=end)[0]
        st_df = pdr.get_data_famafrench('F-F_ST_Reversal_Factor'+dl_freq, start=start, end=end)[0]
        lt_df = pdr.get_data_famafrench('F-F_LT_Reversal_Factor'+dl_freq, start=start, end=end)[0]

        # Concatenate factors as a pandas dataframe
        X = pd.concat([X, mom_df, st_df, lt_df], axis=1) / 100

        if freq == 'weekly' or freq == '_weekly':
            # Convert daily returns to weekly returns
            Y = Y.resample('W-FRI').agg(lambda x: (x + 1).prod() - 1)
            X = X.resample('W-FRI').agg(lambda x: (x + 1).prod() - 1)

        if save_results:
            X.to_pickle('./cache/factor_'+freq+'.pkl')
            Y.to_pickle('./cache/asset_'+freq+'.pkl')

    # Align X and Y to common dates (factors and returns may have different coverage)
    common_dates = X.index.intersection(Y.index)
    X_aligned = X.loc[common_dates]
    Y_aligned = Y.loc[common_dates]

    # Partition dataset into training and testing sets. Lag the data by one observation
    # X[t] is used to predict Y[t+1], avoiding look-ahead bias
    return TrainTest(X_aligned[:-1], n_obs, split), TrainTest(Y_aligned[1:], n_obs, split)

####################################################################################################
# Fetch market data from local CSV files (no network required)
####################################################################################################
def fetch_data_from_disk(start: str, end: str, split: list, freq: str = 'weekly', n_obs: int = 104,
                         n_y: int = None, data_dir: str = './data',
                         use_cache: bool = False, save_results: bool = False):
    """Load market and factor data from local CSV files.

    Reads S&P 500 returns from sp_meta.csv, Fama-French 6 factors from
    factor_panel_long.csv, 5 macro innovations from macro_panel_daily_long.csv,
    and an ESG factor from esg_factor.csv. The effective date range is clamped
    to the data availability window (~2020-01-02 to 2025-12-31).

    Features (X, 12 total):
        FF6  — MKT, SMB, HML, RMW, CMA, MOM
        Macro — DGS10 (Δ10y yield bp), T5YIFR (Δ5y5y breakeven bp),
                DCOILWTICO (WTI log-ret), DHHNGSP (NG log-ret), VIXCLS (ΔVIX)
        ESG  — composite ESG factor return

    Parameters
    ----------
    start : str
        Start date of time series (ISO format, e.g. '2020-01-02').
    end : str
        End date of time series (ISO format, e.g. '2024-12-31').
    split : list
        Train-test split as percentages (must sum to 1).
    freq : str, optional
        Data frequency ('daily' or 'weekly'). The default is 'weekly'.
    n_obs : int, optional
        Number of observations per rolling batch. The default is 104.
    n_y : int, optional
        Number of assets to select (alphabetical). If None, all clean
        tickers are used. The default is None.
    data_dir : str, optional
        Directory containing the data CSVs. The default is './data'.
    use_cache : bool, optional
        Load pre-built pickles from ./cache/ instead of reading CSVs.
        The default is False.
    save_results : bool, optional
        Pickle X and Y to ./cache/ after building. The default is False.

    Returns
    -------
    X : TrainTest
        Feature data split into train and test subsets.
    Y : TrainTest
        Asset return data split into train and test subsets.
    """
    overlap_start, overlap_end = '2020-01-02', '2025-12-31'
    effective_start = max(overlap_start, start)
    effective_end   = min(overlap_end, end)

    if use_cache:
        X = pd.read_pickle('./cache/disk_factor_' + freq + '.pkl')
        Y = pd.read_pickle('./cache/disk_asset_' + freq + '.pkl')
        X = X.loc[effective_start:effective_end]
        Y = Y.loc[effective_start:effective_end]
    else:
        ff_names    = ['MKT-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
        macro_names = ['DGS10', 'T5YIFR', 'DCOILWTICO', 'DHHNGSP', 'VIXCLS']
        final_cols  = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM',
                       'DGS10', 'T5YIFR', 'DCOILWTICO', 'DHHNGSP', 'VIXCLS', 'esg']

        # -- Y: asset returns -------------------------------------------------
        sp = pd.read_csv(data_dir + '/sp_meta.csv', index_col=0, parse_dates=['Date'])
        Y = sp.pivot_table(index='Date', columns='Symbol', values='Adj Close')
        Y.columns.name = None
        Y = Y.pct_change(fill_method=None).dropna(how='all')
        # drop tickers with any NaN in the effective window before resampling
        window = Y.loc[effective_start:effective_end]
        clean  = window.columns[window.isna().sum() == 0]
        Y = Y[clean]
        if n_y is not None:
            Y = Y.iloc[:, :n_y]

        # -- FF factors -------------------------------------------------------
        fp = pd.read_csv(data_dir + '/factor_panel_long.csv', parse_dates=['date'])
        fp = fp[(fp['frequency'] == 'daily') & (fp['factor'].isin(ff_names))]
        ff = fp.pivot_table(index='date', columns='factor', values='value')[ff_names]
        ff.columns.name = None
        ff = ff.rename(columns={'MKT-RF': 'MKT'})

        # -- macro innovations ------------------------------------------------
        mp = pd.read_csv(data_dir + '/macro_panel_daily_long.csv', parse_dates=['date'])
        mp = mp[(mp['kind'] == 'innov') & (mp['series'].isin(macro_names))]
        macro = mp.pivot_table(index='date', columns='series', values='value')[macro_names]
        macro.columns.name = None

        # -- ESG factor -------------------------------------------------------
        esg_raw = pd.read_csv(data_dir + '/esg_factor.csv', index_col=0)
        esg_raw.index = pd.to_datetime(esg_raw['index'])
        esg = esg_raw[['esg']]

        # -- weekly resampling ------------------------------------------------
        if freq == 'weekly':
            compound = lambda x: (x + 1).prod() - 1
            Y     = Y.resample('W-FRI').agg(compound)
            ff_w  = ff.resample('W-FRI').agg(compound)
            esg_w = esg.resample('W-FRI').agg(compound)
            # macro innovations are additive: bp-changes and log-returns both sum
            mac_w = macro.resample('W-FRI').sum(min_count=1)
            X = pd.concat([ff_w, mac_w, esg_w], axis=1)[final_cols]
        else:
            X = pd.concat([ff, macro, esg], axis=1)[final_cols]

        # -- align and clean --------------------------------------------------
        X = X.dropna()
        common = X.index.intersection(Y.index)
        X = X.loc[common]
        Y = Y.loc[common].dropna(axis=1)

        X = X.loc[effective_start:effective_end]
        Y = Y.loc[effective_start:effective_end]

        if save_results:
            X.to_pickle('./cache/disk_factor_' + freq + '.pkl')
            Y.to_pickle('./cache/disk_asset_' + freq + '.pkl')

    # X[t] predicts Y[t+1] — lag by one observation to avoid look-ahead bias
    return TrainTest(X[:-1], n_obs, split), TrainTest(Y[1:], n_obs, split)

####################################################################################################
# Deprecated alias for backward compatibility
####################################################################################################
def AV(*args, **kwargs):
    """Deprecated: Use fetch_market_data() instead.

    This function is kept for backward compatibility with existing code.
    """
    import warnings
    warnings.warn(
        "AV() is deprecated and will be removed in a future version. "
        "Use fetch_market_data() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return fetch_market_data(*args, **kwargs)

####################################################################################################
# stats function
####################################################################################################
def statanalysis(X:pd.DataFrame, Y:pd.DataFrame) -> pd.DataFrame:
    """Conduct a pairwise statistical significance analysis of each feature in X against each asset
    in Y. 

    Parameters
    ----------
    X : pd.DataFrame
        Timeseries of features.
    Y : pd.DataFrame
        Timeseries of asset returns.

    Returns
    -------
    stats : pd.DataFrame
        Table of p-values obtained from regressing each individual feature against each individual 
        asset.

    """
    
    stats = pd.DataFrame(columns=X.columns, index=Y.columns)
    for ticker in Y.columns:
        for feature in X.columns:
            stats.loc[ticker, feature] = sm.OLS(Y[ticker].values, 
                                                sm.add_constant(X[feature]).values
                                                ).fit().pvalues[1]
            
    return stats.astype(float).round(2)
    