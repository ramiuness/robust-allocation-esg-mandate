# Naive Model Module
#
####################################################################################################
## Import libraries
####################################################################################################
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import e2edro.RiskFunctions as rf
import e2edro.PortfolioClasses as pc
import e2edro.e2edro as e2e



####################################################################################################
# CvxpyLayers: Differentiable optimization layers (nominal and distributionally robust)
####################################################################################################
#---------------------------------------------------------------------------------------------------
# base_mod: CvxpyLayer that declares the portfolio optimization problem
#---------------------------------------------------------------------------------------------------
def base_mod(n_y, n_obs, prisk, max_weight=1.0, long_short=False):
    """Base optimization problem declared as a CvxpyLayer object

    Inputs
    n_y: number of assets
    n_obs: Number of scenarios in the dataset
    prisk: Portfolio risk function. Not used in the code but included for the purpose of maintaining the optimization interface consistency.
    max_weight: Maximum weight per asset (default 1.0 = unconstrained long-only).
    long_short: If True, allows short positions (removes nonneg constraint, adds z >= -max_weight).

    Variables
    z: Decision variable. (n_y x 1) vector of decision variables (e.g., portfolio weights)

    Parameters
    y_hat: (n_y x 1) vector of predicted outcomes

    Constraints
    Total budget is equal to 100%, sum(z) == 1
    Long-only by default; long_short=True removes non-negativity and adds symmetric short bound.

    Objective
    Minimize -y_hat @ z
    """
    # Variables
    z = cp.Variable((n_y, 1)) if long_short else cp.Variable((n_y, 1), nonneg=True)

    # Parameters
    y_hat = cp.Parameter(n_y)

    # Constraints
    constraints = [cp.sum(z) == 1]
    if max_weight < 1.0:
        constraints.append(z <= max_weight)
    if long_short:
        constraints.append(z >= -max_weight)

    # Objective function
    objective = cp.Minimize(-y_hat @ z)

    # Construct optimization problem and differentiable layer
    problem = cp.Problem(objective, constraints)

    return CvxpyLayer(problem, parameters=[y_hat], variables=[z])

####################################################################################################
# Naive 'predict-then-optimize'
####################################################################################################
class pred_then_opt(nn.Module):
    """Naive 'predict-then-optimize' portfolio construction module
    """
    def __init__(self, n_x, n_y, n_obs, epsilon=0.5, set_seed=None, prisk='p_var', opt_layer='nominal', cache_path='./cache/', max_weight=1.0, long_short=False):
        """Naive 'predict-then-optimize' portfolio construction module

        This NN module implements a linear prediction layer 'pred_layer' and an optimization layer 
        'opt_layer'. The model is 'naive' since it optimizes each layer separately. 

        Inputs
        n_x: Number of inputs (i.e., features) in the prediction model
        n_y: Number of outputs from the prediction model
        n_obs: Number of scenarios from which to calculate the sample set of residuals
        prisk: String. Portfolio risk function. Used in the opt_layer
        
        Output
        pred_then_opt: nn.Module object 
        """
        super(pred_then_opt, self).__init__()

        if set_seed is not None:
            torch.manual_seed(set_seed)
            self.seed = set_seed

        self.n_x = n_x
        self.n_y = n_y
        self.n_obs = n_obs
        self.max_weight = max_weight  # Max weight per asset for diversification
        self.long_short = long_short  # Allow short positions if True
        self.perf_period = 13

        # Store epsilon and prisk for layer rebuild capability
        self.epsilon_fixed = epsilon
        self.prisk_func = eval('rf.'+prisk)

        # Register 'gamma' (risk-return trade-off parameter)
        # self.gamma = nn.Parameter(torch.FloatTensor(1).uniform_(0.037, 0.173))
        self.gamma = nn.Parameter(torch.FloatTensor(1).uniform_(0.02, 0.1))
        self.gamma.requires_grad = False

        # Record the model design: nominal, base, base_rom or DRO
        if opt_layer == 'nominal':
            self.model_type = 'nom'
        elif opt_layer == 'base_mod':
            self.model_type = 'base_mod'
        elif opt_layer == 'base_rom':
            self.model_type = 'base_rom'
            self.epsilon = nn.Parameter(torch.FloatTensor([epsilon]))
            self.epsilon.requires_grad = False
        else:
            # Register 'delta' (ambiguity sizing parameter) for DRO model
            if opt_layer == 'hellinger':
                ub = (1 - 1/(n_obs**0.5)) / 2
                lb = (1 - 1/(n_obs**0.5)) / 10
            else:
                ub = (1 - 1/n_obs) / 2
                lb = (1 - 1/n_obs) / 10
            self.delta = nn.Parameter(torch.FloatTensor(1).uniform_(lb, ub))
            self.delta.requires_grad = False
            self.model_type = 'dro' 

        # LAYER: OLS linear prediction
        self.pred_layer = nn.Linear(n_x, n_y)
        self.pred_layer.weight.requires_grad = False
        self.pred_layer.bias.requires_grad = False
        

        # LAYER: Optimization model
        if opt_layer == 'base_mod':
            self.opt_layer = base_mod(n_y, n_obs, eval('rf.'+prisk),
                                      max_weight=max_weight, long_short=long_short)
        elif opt_layer == 'base_rom':
            placeholder = np.eye(n_y)
            self.sigma_mu_hat = placeholder
            self.opt_layer = e2e.base_rom(n_y, n_obs, eval('rf.'+prisk), placeholder,
                                          max_weight=max_weight, long_short=long_short)
        else:
            self.opt_layer = eval(opt_layer)(n_y, n_obs, eval('rf.'+prisk),
                                             max_weight=max_weight, long_short=long_short)
        # Store reference path to store model data
        self.cache_path = cache_path

    #-----------------------------------------------------------------------------------------------
    # forward: forward pass of the e2e neural net
    #-----------------------------------------------------------------------------------------------
    def forward(self, X, Y):
        """Forward pass of the predict-then-optimize module

        The inputs 'X' are passed through the prediction layer to yield predictions 'Y_hat'. The
        residuals from prediction are then calcuclated as 'ep = Y - Y_hat'. Finally, the residuals
        are passed to the optimization layer to find the optimal decision z_star.

        Inputs
        X: Features. ([n_obs+1] x n_x) torch tensor with feature timeseries data
        Y: Realizations. (n_obs x n_y) torch tensor with asset timeseries data

        Other 
        ep: Residuals. (n_obs x n_y) matrix of the residual between realizations and predictions

        Outputs
        y_hat: Prediction. (n_y x 1) vector of outputs of the prediction layer
        z_star: Optimal solution. (n_y x 1) vector of asset weights
        """
        # Predict y_hat from x
        Y_hat = torch.stack([self.pred_layer(x_t) for x_t in X])

        # Calculate residuals and process them
        ep = Y - Y_hat[:-1]
        y_hat = Y_hat[-1]

        # Optimization solver arguments (from CVXPY for SCS solver)
        solver_args = {'solve_method': 'ECOS'}

        # Optimize z per scenario
        # Determine whether nominal or dro model
        if self.model_type == 'nom':
            z_star, = self.opt_layer(ep, y_hat, self.gamma, solver_args=solver_args)
        elif self.model_type == 'dro':
            z_star, = self.opt_layer(ep, y_hat, self.gamma, self.delta, solver_args=solver_args)
        elif self.model_type == 'base_mod':
            z_star, = self.opt_layer(y_hat, solver_args=solver_args)
        elif self.model_type == 'base_rom':
            z_star, = self.opt_layer(y_hat, self.epsilon, solver_args=solver_args)

        return z_star, y_hat

    #-----------------------------------------------------------------------------------------------
    # net_test: Test the e2e neural net
    #-----------------------------------------------------------------------------------------------
    def net_roll_test(self, X, Y, n_roll=4):
        """Neural net rolling window out-of-sample test

        Inputs
        X: Features. ([n_obs+1] x n_x) torch tensor with feature timeseries data
        Y: Realizations. (n_obs x n_y) torch tensor with asset timeseries data
        n_roll: Number of training periods (i.e., number of times to retrain the model)

        Output
        self.portfolio: add the backtest results to the e2e_net object
        """

        # Declare backtest object to hold the test results
        portfolio = pc.backtest(len(Y.test())-Y.n_obs, self.n_y, Y.test().index[Y.n_obs:])

        # Store initial train/test split
        init_split = Y.split

        # Window size
        win_size = init_split[1] / n_roll

        split = [0, 0]
        t = 0
        for i in range(n_roll):

            print(f"Out-of-sample window: {i+1} / {n_roll}")

            split[0] = init_split[0] + win_size * i
            if i < n_roll-1:
                split[1] = win_size
            else:
                split[1] = 1 - split[0]

            X.split_update(split), Y.split_update(split)

            test_set = DataLoader(pc.SlidingWindow(X.test(), Y.test(), self.n_obs, 1))

            X_train, Y_train = X.train().copy(), Y.train()
            X_train.insert(0,'ones', 1.0)

            X_train = Variable(torch.tensor(X_train.values, dtype=torch.double))
            Y_train = Variable(torch.tensor(Y_train.values, dtype=torch.double))

            Theta = torch.linalg.lstsq(X_train, Y_train).solution.T
            del X_train, Y_train

            with torch.no_grad():
                self.pred_layer.bias.copy_(Theta[:,0])
                self.pred_layer.weight.copy_(Theta[:,1:])

            # Update Sigma_mu_hat for base_rom using OLS-initialised B and current window's Cov(x)
            if self.model_type == 'base_rom':
                diag_info = self.update_sigma_mu_hat(X.train())
                if diag_info.get('updated', False):
                    print(f"  Sigma_mu_hat updated (trace: {diag_info['sigma_mu_hat_trace']:.2e})")

            # Test model
            with torch.no_grad():
                for j, (x, y, y_perf) in enumerate(test_set):
                
                    # Predict and optimize
                    z_star, _ = self(x.squeeze(), y.squeeze())

                    # Store portfolio weights and returns for each time step 't'
                    portfolio.weights[t] = z_star.squeeze()
                    portfolio.rets[t] = y_perf.squeeze() @ portfolio.weights[t]

                    t += 1

        # Reset dataset
        X, Y = X.split_update(init_split), Y.split_update(init_split)

        # Calculate the portfolio statistics using the realized portfolio returns
        portfolio.stats()

        self.portfolio = portfolio

    #-----------------------------------------------------------------------------------------------
    # update_sigma_mu_hat: Update estimator covariance and rebuild base_rom layer
    #-----------------------------------------------------------------------------------------------
    def update_sigma_mu_hat(self, X_train):
        """Recompute Sigma_mu_hat = B Cov(x) B^T from current OLS weights and rebuild opt_layer.

        Called once per roll window after OLS initialisation. B is fixed in pred_then_opt
        (no gradient learning), so a single update per roll is sufficient.

        Parameters
        ----------
        X_train : pd.DataFrame
            Factor data for the current training window (without a ones column).

        Returns
        -------
        dict
            Diagnostic information (updated, sigma_mu_hat_trace).
        """
        diagnostics = {}
        if self.model_type != 'base_rom':
            diagnostics['updated'] = False
            diagnostics['reason'] = f'Model type is {self.model_type}, not base_rom'
            return diagnostics

        B     = self.pred_layer.weight.detach().numpy()   # (n_y, n_x)
        cov_x = X_train.cov().values                      # (n_x, n_x)
        sigma_mu_hat_new = B @ cov_x @ B.T
        self.sigma_mu_hat = sigma_mu_hat_new

        self.opt_layer = e2e.base_rom(
            self.n_y, self.n_obs, self.prisk_func,
            sigma_mu_hat_new, self.max_weight, long_short=self.long_short
        )

        diagnostics['updated'] = True
        diagnostics['sigma_mu_hat_trace'] = float(np.trace(sigma_mu_hat_new))
        return diagnostics

####################################################################################################
# Equal weight
####################################################################################################
class equal_weight:
    """Naive 'equally-weighted' portfolio construction module
    """
    def __init__(self, n_x, n_y, n_obs):
        """Naive 'equally-weighted' portfolio construction module

        This object implements a basic equally-weighted investment strategy.

        Inputs
        n_x: Number of inputs (i.e., features) in the prediction model
        n_y: Number of outputs from the prediction model
        n_obs: Number of scenarios from which to calculate the sample set of residuals
        """
        self.n_x = n_x
        self.n_y = n_y
        self.n_obs = n_obs
        self.perf_period = 13
    
    #-----------------------------------------------------------------------------------------------
    # net_test: Test the e2e neural net
    #-----------------------------------------------------------------------------------------------
    def net_roll_test(self, X, Y, n_roll=4):
        """Neural net rolling window out-of-sample test

        Inputs
        X: Features. ([n_obs+1] x n_x) torch tensor with feature timeseries data
        Y: Realizations. (n_obs x n_y) torch tensor with asset timeseries data
        n_roll: Number of training periods (i.e., number of times to retrain the model)

        Output 
        self.portfolio: add the backtest results to the e2e_net object
        """

        # Declare backtest object to hold the test results
        portfolio = pc.backtest(len(Y.test())-Y.n_obs, self.n_y, Y.test().index[Y.n_obs:])

        test_set = DataLoader(pc.SlidingWindow(X.test(), Y.test(), self.n_obs, 1))

        # Test model
        t = 0
        for j, (x, y, y_perf) in enumerate(test_set):
            
            portfolio.weights[t] = np.ones(self.n_y) / self.n_y
            portfolio.rets[t] = y_perf.squeeze() @ portfolio.weights[t]
            t += 1

        # Calculate the portfolio statistics using the realized portfolio returns
        portfolio.stats()

        self.portfolio = portfolio

####################################################################################################
# Find gamma range
####################################################################################################
class gamma_range(nn.Module):
    """Simple way to approximately determine the appropriate values of gamma
    """
    def __init__(self, n_x, n_y, n_obs):
        """Naive 'predict-then-optimize' portfolio construction module

        This NN module implements a linear prediction layer 'pred_layer' and an optimization layer 
        'opt_layer'. The model is 'naive' since it optimizes each layer separately. 

        Inputs
        n_x: Number of inputs (i.e., features) in the prediction model
        n_y: Number of outputs from the prediction model
        n_obs: Number of scenarios from which to calculate the sample set of residuals
        prisk: String. Portfolio risk function. Used in the opt_layer
        
        Output
        pred_then_opt: nn.Module object 
        """
        super(gamma_range, self).__init__()

        self.n_x = n_x
        self.n_y = n_y
        self.n_obs = n_obs

        # LAYER: OLS linear prediction
        self.pred_layer = nn.Linear(n_x, n_y)
        self.pred_layer.weight.requires_grad = False
        self.pred_layer.bias.requires_grad = False

    #-----------------------------------------------------------------------------------------------
    # forward: forward pass of the e2e neural net
    #-----------------------------------------------------------------------------------------------
    def forward(self, X, Y):
        """Forward pass of the predict-then-optimize module

        The inputs 'X' are passed through the prediction layer to yield predictions 'Y_hat'. The
        residuals from prediction are then calcuclated as 'ep = Y - Y_hat'. Finally, the residuals
        are passed to the optimization layer to find the optimal decision z_star.

        Inputs
        X: Features. ([n_obs+1] x n_x) torch tensor with feature timeseries data
        Y: Realizations. (n_obs x n_y) torch tensor with asset timeseries data

        Other 
        ep: Residuals. (n_obs x n_y) matrix of the residual between realizations and predictions

        Outputs
        y_hat: Prediction. (n_y x 1) vector of outputs of the prediction layer
        z_star: Optimal solution. (n_y x 1) vector of asset weights
        """
        # Predict y_hat from x
        Y_hat = torch.stack([self.pred_layer(x_t) for x_t in X])

        # Calculate residuals and process them
        ep = Y - Y_hat[:-1]
        cov_ep = torch.cov(ep.T)

        # Find prediction
        y_hat = Y_hat[-1]

        # Set z=1/n per scenario
        z_star = torch.ones(self.n_y, dtype=torch.double) / self.n_y

        gamma = ((z_star.T @ cov_ep) @ z_star) / torch.abs(y_hat @ z_star)

        return gamma

    #-----------------------------------------------------------------------------------------------
    # gamma_eval: Find the range of gamma
    #-----------------------------------------------------------------------------------------------
    def gamma_eval(self, X, Y):
        """Use the equal weight portfolio and the nominal distribution to find appropriate
        values of gamma. 

        Inputs
        X: Features. ([n_obs+1] x n_x) torch tensor with feature timeseries data
        Y: Realizations. (n_obs x n_y) torch tensor with asset timeseries data

        Output
        gamma: estimated gamma valules for each observation in the training set
        """

        # Initialize the prediction layer weights to OLS regression weights
        X_train, Y_train = X.train().copy(), Y.train()
        X_train.insert(0,'ones', 1.0)

        X_train = Variable(torch.tensor(X_train.values, dtype=torch.double))
        Y_train = Variable(torch.tensor(Y_train.values, dtype=torch.double))

        Theta = torch.linalg.lstsq(X_train, Y_train).solution.T
        del X_train, Y_train

        with torch.no_grad():
            self.pred_layer.bias.copy_(Theta[:,0])
            self.pred_layer.weight.copy_(Theta[:,1:])

       # Construct training and validation DataLoader objects
        train_set = DataLoader(pc.SlidingWindow(X.train(), Y.train(), self.n_obs, 0))

        # Test model
        with torch.no_grad():
            gamma = []
            for t, (x, y, y_perf) in enumerate(train_set):
                gamma.append(self(x.squeeze(),y.squeeze()))

        return gamma
