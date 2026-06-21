# E2E DRO Module
#
####################################################################################################
## Import libraries
####################################################################################################
import os
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import e2edro.RiskFunctions as rf
import e2edro.LossFunctions as lf
import e2edro.PortfolioClasses as pc
import e2edro.DataLoad as dl

import psutil
num_cores = psutil.cpu_count()
torch.set_num_threads(num_cores)
if psutil.MACOS:
    num_cores = 0

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

#---------------------------------------------------------------------------------------------------
# nominal: CvxpyLayer that declares the portfolio optimization problem
#---------------------------------------------------------------------------------------------------
def nominal(n_y, n_obs, prisk, max_weight=1.0, long_short=False):
    """Nominal optimization problem declared as a CvxpyLayer object

    Inputs
    n_y: number of assets
    n_obs: Number of scenarios in the dataset
    prisk: Portfolio risk function
    max_weight: Maximum weight per asset (default 1.0 = unconstrained long-only).
    long_short: If True, allows short positions.

    Variables
    z: Decision variable. (n_y x 1) vector of decision variables (e.g., portfolio weights)
    c_aux: Auxiliary Variable. Scalar
    obj_aux: Auxiliary Variable. (n_obs x 1) vector. Allows for a tractable DR counterpart.
    mu_aux: Auxiliary Variable. Scalar. Represents the portfolio conditional expected return.

    Parameters
    ep: (n_obs x n_y) matrix of residuals
    y_hat: (n_y x 1) vector of predicted outcomes (e.g., conditional expected returns)
    gamma: Scalar. Trade-off between conditional expected return and model error.

    Constraints
    Total budget is equal to 100%, sum(z) == 1
    Long-only by default; long_short=True removes non-negativity and adds symmetric short bound.

    Objective
    Minimize (1/n_obs) * cp.sum(obj_aux) - gamma * mu_aux
    """
    # Variables
    z = cp.Variable((n_y, 1)) if long_short else cp.Variable((n_y, 1), nonneg=True)
    c_aux = cp.Variable()
    obj_aux = cp.Variable(n_obs)
    mu_aux = cp.Variable()

    # Parameters
    ep = cp.Parameter((n_obs, n_y))
    y_hat = cp.Parameter(n_y)
    gamma = cp.Parameter(nonneg=True)

    # Constraints
    constraints = [cp.sum(z) == 1, mu_aux == y_hat @ z]
    if max_weight < 1.0:
        constraints.append(z <= max_weight)
    if long_short:
        constraints.append(z >= -max_weight)
    for i in range(n_obs):
        constraints += [obj_aux[i] >= prisk(z, c_aux, ep[i])]

    # Objective function
    objective = cp.Minimize((1/n_obs) * cp.sum(obj_aux) - gamma * mu_aux)

    # Construct optimization problem and differentiable layer
    problem = cp.Problem(objective, constraints)

    return CvxpyLayer(problem, parameters=[ep, y_hat, gamma], variables=[z])

#---------------------------------------------------------------------------------------------------
# Total Variation: sum_t abs(p_t - q_t) <= delta
#---------------------------------------------------------------------------------------------------
def tv(n_y, n_obs, prisk, max_weight=1.0, long_short=False):
    """DRO layer using the 'Total Variation' distance to define the probability ambiguity set.
    From Ben-Tal et al. (2013).
    Total Variation: sum_t abs(p_t - q_t) <= delta

    Inputs
    n_y: Number of assets
    n_obs: Number of scenarios in the dataset
    prisk: Portfolio risk function
    max_weight: Maximum weight per asset (default 1.0 = unconstrained long-only).
    long_short: If True, allows short positions.

    Variables
    z: Decision variable. (n_y x 1) vector of decision variables (e.g., portfolio weights)
    c_aux: Auxiliary Variable. Scalar. Allows us to p-linearize the derivation of the variance
    lambda_aux: Auxiliary Variable. Scalar. Allows for a tractable DR counterpart.
    eta_aux: Auxiliary Variable. Scalar. Allows for a tractable DR counterpart.
    obj_aux: Auxiliary Variable. (n_obs x 1) vector. Allows for a tractable DR counterpart.

    Parameters
    ep: (n_obs x n_y) matrix of residuals
    y_hat: (n_y x 1) vector of predicted outcomes (e.g., conditional expected returns)
    delta: Scalar. Maximum distance between p and q.
    gamma: Scalar. Trade-off between conditional expected return and model error.
    mu_aux: Auxiliary Variable. Scalar. Represents the portfolio conditional expected return.

    Constraints
    Total budget is equal to 100%, sum(z) == 1
    Long-only by default; long_short=True removes non-negativity and adds symmetric short bound.

    Objective
    Minimize eta_aux + delta * lambda_aux + (1/n_obs) * sum(beta_aux) - gamma * y_hat @ z
    """

    # Variables
    z = cp.Variable((n_y, 1)) if long_short else cp.Variable((n_y, 1), nonneg=True)
    c_aux = cp.Variable()
    lambda_aux = cp.Variable(nonneg=True)
    eta_aux = cp.Variable()
    beta_aux = cp.Variable(n_obs)
    mu_aux = cp.Variable()

    # Parameters
    ep = cp.Parameter((n_obs, n_y))
    y_hat = cp.Parameter(n_y)
    gamma = cp.Parameter(nonneg=True)
    delta = cp.Parameter(nonneg=True)

    # Constraints
    constraints = [cp.sum(z) == 1, beta_aux >= -lambda_aux, mu_aux == y_hat @ z]
    if max_weight < 1.0:
        constraints.append(z <= max_weight)
    if long_short:
        constraints.append(z >= -max_weight)
    for i in range(n_obs):
        constraints += [beta_aux[i] >= prisk(z, c_aux, ep[i]) - eta_aux]
        constraints += [lambda_aux >= prisk(z, c_aux, ep[i]) - eta_aux]

    # Objective function
    objective = cp.Minimize(eta_aux + delta * lambda_aux + (1/n_obs) * cp.sum(beta_aux)
                            - gamma * mu_aux)

    # Construct optimization problem and differentiable layer
    problem = cp.Problem(objective, constraints)

    return CvxpyLayer(problem, parameters=[ep, y_hat, gamma, delta], variables=[z])

#---------------------------------------------------------------------------------------------------
# Hellinger distance: sum_t (sqrt(p_t) - sqrtq_t))^2 <= delta
#---------------------------------------------------------------------------------------------------
def hellinger(n_y, n_obs, prisk, max_weight=1.0, long_short=False):
    """DRO layer using the Hellinger distance to define the probability ambiguity set.
    from Ben-Tal et al. (2013).
    Hellinger distance: sum_t (sqrt(p_t) - sqrtq_t))^2 <= delta

    Inputs
    n_y: number of assets
    n_obs: Number of scenarios in the dataset
    prisk: Portfolio risk function
    max_weight: Maximum weight per asset (default 1.0 = unconstrained long-only).
    long_short: If True, allows short positions.

    Variables
    z: Decision variable. (n_y x 1) vector of decision variables (e.g., portfolio weights)
    c_aux: Auxiliary Variable. Scalar. Allows us to p-linearize the derivation of the variance
    lambda_aux: Auxiliary Variable. Scalar. Allows for a tractable DR counterpart.
    xi_aux: Auxiliary Variable. Scalar. Allows for a tractable DR counterpart.
    beta_aux: Auxiliary Variable. (n_obs x 1) vector. Allows for a tractable DR counterpart.
    s_aux: Auxiliary Variable. (n_obs x 1) vector. Allows for a tractable SOC constraint.
    mu_aux: Auxiliary Variable. Scalar. Represents the portfolio conditional expected return.

    Parameters
    ep: (n_obs x n_y) matrix of residuals
    y_hat: (n_y x 1) vector of predicted outcomes (e.g., conditional expected returns)
    delta: Scalar. Maximum distance between p and q.
    gamma: Scalar. Trade-off between conditional expected return and model error.

    Constraints
    Total budget is equal to 100%, sum(z) == 1
    Long-only by default; long_short=True removes non-negativity and adds symmetric short bound.

    Objective
    Minimize xi_aux + (delta-1) * lambda_aux + (1/n_obs) * sum(beta_aux) - gamma * y_hat @ z
    """

    # Variables
    z = cp.Variable((n_y, 1)) if long_short else cp.Variable((n_y, 1), nonneg=True)
    c_aux = cp.Variable()
    lambda_aux = cp.Variable(nonneg=True)
    xi_aux = cp.Variable()
    beta_aux = cp.Variable(n_obs, nonneg=True)
    tau_aux = cp.Variable(n_obs, nonneg=True)
    mu_aux = cp.Variable()

    # Parameters
    ep = cp.Parameter((n_obs, n_y))
    y_hat = cp.Parameter(n_y)
    gamma = cp.Parameter(nonneg=True)
    delta = cp.Parameter(nonneg=True)

    # Constraints
    constraints = [cp.sum(z) == 1, mu_aux == y_hat @ z]
    if max_weight < 1.0:
        constraints.append(z <= max_weight)
    if long_short:
        constraints.append(z >= -max_weight)
    for i in range(n_obs):
        constraints += [xi_aux + lambda_aux >= prisk(z, c_aux, ep[i]) + tau_aux[i]]
        constraints += [beta_aux[i] >= cp.quad_over_lin(lambda_aux, tau_aux[i])]

    # Objective function
    objective = cp.Minimize(xi_aux + (delta-1) * lambda_aux + (1/n_obs) * cp.sum(beta_aux)
                            - gamma * mu_aux)

    # Construct optimization problem and differentiable layer
    problem = cp.Problem(objective, constraints)

    return CvxpyLayer(problem, parameters=[ep, y_hat, gamma, delta], variables=[z])

####################################################################################################
# base_rom: Estimation-robust layer (ellipsoidal uncertainty on μ̂)
####################################################################################################
def base_rom(n_y, n_obs, prisk, sigma_mu_hat, max_weight=1.0, long_short=False):
    """Estimation-robust SOCP layer.

    Reformulates min_w max_{μ ∈ U(ε)} -μᵀw into the tractable SOCP:
        min_z  -y_hat @ z + epsilon * ||L_thin.T @ z||_2
    where L_thin is the thin factor of sigma_mu_hat = B Cov(x) Bᵀ.

    sigma_mu_hat is always rank-deficient (rank ≤ n_x < n_y) because
    rank(B Cov(x) Bᵀ) ≤ rank(B) ≤ n_x. Thin eigendecomposition handles
    this exactly without any ridge perturbation.

    Parameters
    n_y: Number of assets
    n_obs: Number of scenarios (accepted for interface consistency, not used)
    prisk: Risk function (accepted for interface consistency, not used)
    sigma_mu_hat: (n_y x n_y) ndarray. Estimator covariance Σ_{μ̂} = B Cov(x) Bᵀ
    max_weight: Maximum weight per asset (default 1.0 = unconstrained long-only).
    long_short: If True, allows short positions (removes nonneg, adds z >= -max_weight).

    CvxpyLayer parameters: [y_hat, epsilon]
    """
    # Thin eigendecomposition: retain eigenvectors above relative threshold
    eigvals, eigvecs = np.linalg.eigh(np.array(sigma_mu_hat))   # ascending order
    tol = 1e-10 * eigvals[-1]
    mask = eigvals > tol
    L_thin = eigvecs[:, mask] @ np.diag(np.sqrt(eigvals[mask]))  # (n_y, r), r <= n_x

    z       = cp.Variable((n_y, 1)) if long_short else cp.Variable((n_y, 1), nonneg=True)
    y_hat   = cp.Parameter(n_y)
    epsilon = cp.Parameter(nonneg=True)

    constraints = [cp.sum(z) == 1]
    if max_weight < 1.0:
        if max_weight * n_y < 1.0:
            raise ValueError(
                f"Infeasible: max_weight={max_weight} with n_y={n_y} assets. "
                f"Need max_weight >= {1.0/n_y:.4f} (= 1/n_y) for feasibility."
            )
        constraints.append(z <= max_weight)
    if long_short:
        constraints.append(z >= -max_weight)

    # L_thin.T @ z is (r, 1) — affine in z (L_thin.T is a numpy constant) → DPP-compliant
    # epsilon is cp.Parameter(nonneg=True) multiplying a convex norm → DPP-compliant
    objective = cp.Minimize(-y_hat @ z + epsilon * cp.norm(L_thin.T @ z, 2))
    problem   = cp.Problem(objective, constraints)

    return CvxpyLayer(problem, parameters=[y_hat, epsilon], variables=[z])

####################################################################################################
# E2E neural network module
####################################################################################################
class DeviceDataLoader:
    """GPU MOD: Wrap DataLoader to move batches to GPU
    """
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device

    def __iter__(self):
        for x, y, y_perf in self.loader:
            yield x.to(self.device), y.to(self.device), y_perf.to(self.device)

    def __len__(self):
        return len(self.loader)


class e2e_net(nn.Module):
    """End-to-end DRO learning neural net module.
    """
    def __init__(self, n_x, n_y, n_obs, opt_layer='nominal', prisk='p_var', perf_loss='sharpe_loss',
                pred_model='linear', pred_loss_factor=0.5, perf_period=13, train_pred=True, train_gamma=True, train_delta=True, train_epsilon=True, set_seed=None, epochs=10, lr=1e-3, epsilon_lr=None, weight_decay=0.0, gamma_lr=None, long_short=False, cache_path='./cache/', max_weight=None):
        """End-to-end learning neural net module

        This NN module implements a linear prediction layer 'pred_layer' and a DRO layer 
        'opt_layer' based on a tractable convex formulation from Ben-Tal et al. (2013). 'delta' and
        'gamma' are declared as nn.Parameters so that they can be 'learned'.

        Inputs
        n_x: Number of inputs (i.e., features) in the prediction model
        n_y: Number of outputs from the prediction model
        n_obs: Number of scenarios from which to calculate the sample set of residuals
        sigma: Covariance matrix  of the returns
        prisk: String. Portfolio risk function. Used in the opt_layer
        opt_layer: String. Determines which CvxpyLayer-object to call for the optimization layer
        perf_loss: Performance loss function based on out-of-sample financial performance
        pred_loss_factor: Trade-off between prediction loss function and performance loss function.
            Set 'pred_loss_factor=None' to define the loss function purely as 'perf_loss'
        perf_period: Number of lookahead realizations used in 'perf_loss()'
        train_pred: Boolean. Choose if the prediction layer is learnable (or keep it fixed)
        train_gamma: Boolean. Choose if the risk appetite parameter gamma is learnable
        train_delta: Boolean. Choose if the robustness parameter delta is learnable
        set_seed: (Optional) Int. Set the random seed for replicability

        Output
        e2e_net: nn.Module object 
        """
        super(e2e_net, self).__init__()
        self.double()

        # Set random seed (to be used for replicability of numerical experiments)
        if set_seed is not None:
            torch.manual_seed(set_seed)
            self.seed = set_seed

        self.n_x = n_x
        self.n_y = n_y
        self.n_obs = n_obs
        self.max_weight = max_weight  # Max weight per asset for diversification
        self.epochs = epochs  #it seems that i have to add it there is a call to self.epochs in train_net()
        self.lr = lr  #it seems that i have to add it there is a call to self.lr in train_net()
        self.epsilon_lr = epsilon_lr  # Separate learning rate for epsilon (if None, uses lr)
        self.weight_decay = weight_decay  # L2 regularization on prediction weights only
        self.gamma_lr = gamma_lr          # Separate learning rate for gamma/delta (portfolio params)
        self.long_short = long_short      # Allow short positions if True

        # Store prisk for layer rebuild capability (used by base_rom)
        self.prisk_func = eval('rf.'+prisk)
        # Prediction loss function
        if pred_loss_factor is not None:
            self.pred_loss_factor = pred_loss_factor
            self.pred_loss = torch.nn.MSELoss()
        else:
            self.pred_loss = None

        # Define performance loss
        self.perf_loss = eval('lf.'+perf_loss)

        # Number of time steps to evaluate the task loss
        self.perf_period = perf_period

        # Register 'gamma' (modeling risk-return trade-off parameter)
        self.gamma = nn.Parameter(torch.FloatTensor(1).uniform_(0.02, 0.1))
        self.gamma.requires_grad = train_gamma
        self.gamma_init = self.gamma.item()

        # Record the model design: nominal, base or DRO
        if opt_layer == 'nominal':
            self.model_type = 'nom'
        elif opt_layer == 'base_mod':
            self.gamma.requires_grad = False
            self.model_type = 'base_mod'
        elif opt_layer == 'base_rom':
            if pred_model != 'linear':
                raise ValueError(
                    "opt_layer='base_rom' requires pred_model='linear'. "
                    "Sigma_mu_hat = B Cov(x) B^T is only defined for a single factor-loading matrix B."
                )
            self.gamma.requires_grad = False
            self.epsilon = nn.Parameter(torch.FloatTensor(1).uniform_(0.1, 1.0))
            self.epsilon.requires_grad = train_epsilon
            self.epsilon_init = self.epsilon.item()
            self.model_type = 'base_rom'
        else:
            # Register 'delta' (ambiguity sizing parameter) for DR layer
            if opt_layer == 'hellinger':
                ub = (1 - 1/(n_obs**0.5)) / 2
                lb = (1 - 1/(n_obs**0.5)) / 10
            else:
                ub = (1 - 1/n_obs) / 2
                lb = (1 - 1/n_obs) / 10
            self.delta = nn.Parameter(torch.FloatTensor(1).uniform_(lb, ub))
            self.delta.requires_grad = train_delta
            self.delta_init = self.delta.item()
            self.model_type = 'dro'

        # LAYER: Prediction model
        self.pred_model = pred_model
        if pred_model == 'linear':
            # Linear prediction model
            self.pred_layer = nn.Linear(n_x, n_y)
            self.pred_layer.weight.requires_grad = train_pred
            self.pred_layer.bias.requires_grad = train_pred
        elif pred_model == '2layer':
            # Neural net with 2 hidden layers 
            self.pred_layer = nn.Sequential(nn.Linear(n_x, int(0.5*(n_x+n_y))),
                      nn.ReLU(),
                      nn.Linear(int(0.5*(n_x+n_y)), n_y),
                      nn.ReLU(),
                      nn.Linear(n_y, n_y))
        elif pred_model == '3layer':
            # Neural net with 3 hidden layers 
            self.pred_layer = nn.Sequential(nn.Linear(n_x, int(0.5*(n_x+n_y))),
                      nn.ReLU(),
                      nn.Linear(int(0.5*(n_x+n_y)), int(0.6*(n_x+n_y))),
                      nn.ReLU(),
                      nn.Linear(int(0.6*(n_x+n_y)), n_y),
                      nn.ReLU(),
                      nn.Linear(n_y, n_y))

        # LAYER: Optimization model
        if opt_layer == 'base_rom':
            placeholder = np.eye(n_y)
            self.sigma_mu_hat = placeholder
            self._cov_x_cache = None  # populated by update_sigma_mu_hat before net_train
            self.opt_layer = base_rom(n_y, n_obs, eval('rf.'+prisk), placeholder,
                                      max_weight=max_weight, long_short=long_short)
        else:
            self.opt_layer = eval(opt_layer)(n_y, n_obs, eval('rf.'+prisk),
                                             max_weight=max_weight, long_short=long_short)
        # Store reference path to store model data
        self.cache_path = cache_path

        # Store initial model. During every rolling-window, back-test, or cross-validation fold, the code needs to reset the network to a clean "initial" state before retraining on the new window.
        if self.model_type == 'base_mod':
            self.init_state_path = (
                        cache_path
                        + self.model_type
                        + '_initial_state_'
                        + pred_model
                    )
        elif self.model_type == 'base_rom':
        # Estimation-robust layer: epsilon may or may not be learnable
            self.init_state_path = (
                        cache_path
                        + self.model_type
                        + '_initial_state_'
                        + pred_model
                        + '_TrainEpsilon'
                        + str(train_epsilon)
                    )
        elif train_gamma and train_delta:
            self.init_state_path = cache_path + self.model_type+'_initial_state_' + pred_model
        elif train_delta and not train_gamma:
            self.init_state_path = cache_path + self.model_type+'_initial_state_' + pred_model + '_TrainGamma'+str(train_gamma)
        elif train_gamma and not train_delta:
            self.init_state_path = cache_path + self.model_type+'_initial_state_' + pred_model + '_TrainDelta'+str(train_delta)
        elif not train_gamma and not train_delta:
            self.init_state_path = cache_path + self.model_type+'_initial_state_' + pred_model + '_TrainGamma'+str(train_gamma) + '_TrainDelta'+str(train_delta)
        # Store a checkpoint of the just-initialised weights. Restores the pristine state before each new roll (such as in net_roll_test
  # Make sure this is imported at the top

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(self.init_state_path), exist_ok=True)

        
        torch.save(self.state_dict(), self.init_state_path)

    #-----------------------------------------------------------------------------------------------
    # calibrate_pred_loss_factor: balance loss scales at OLS initialization
    #-----------------------------------------------------------------------------------------------
    def calibrate_pred_loss_factor(self, X_train, Y_train, target_ratio=0.5):
        """Set pred_loss_factor so the prediction co-objective has `target_ratio` weight
        relative to the performance loss at the current (OLS) initialization.

        Runs one no-grad forward pass on the training data, then updates self.pred_loss_factor.
        Returns the calibrated value (or None if pred_loss is disabled).

        target_ratio: fraction of the performance loss magnitude assigned to the prediction
            term at initialization. E.g. 0.5 means prediction loss gets half the gradient
            weight of the task loss at the start of training.
        """
        if self.pred_loss is None:
            return None
        loader = DataLoader(pc.SlidingWindow(X_train, Y_train, self.n_obs, self.perf_period))
        self.eval()
        with torch.no_grad():
            x, y, y_perf = next(iter(loader))
            z_star, y_hat = self(x.squeeze(), y.squeeze())
            perf_l = abs(self.perf_loss(z_star, y_perf.squeeze()).item())
            pred_l = abs(self.pred_loss(y_hat, y_perf.squeeze()[0]).item()) / self.n_y
        if pred_l > 0:
            self.pred_loss_factor = target_ratio * perf_l / pred_l
        return self.pred_loss_factor

    #-----------------------------------------------------------------------------------------------
    # forward: forward pass of the e2e neural net
    #-----------------------------------------------------------------------------------------------
    def forward(self, X, Y):
        """Forward pass of the NN module

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
        # Multiple predictions Y_hat from X
        Y_hat = torch.stack([self.pred_layer(x_t) for x_t in X])
        
        # Calculate residuals and process them
        if Y.shape[0] == Y_hat.shape[0]: # I had to add this step to get e2e_net to output! Better if revised!
            ep = Y - Y_hat
            y_hat = Y_hat[-1]
        else:
            ep = Y - Y_hat[:-1]
            y_hat = Y_hat[-1]

        # Optimization solver arguments (from CVXPY for ECOS/SCS solver)
        solver_args = {'solve_method': 'ECOS', 'max_iters': 250, 'abstol': 1e-7}
        #solver_args = {'solve_method': 'SCS', 'eps': 1e-7, 'acceleration_lookback': 5, 'max_iters':20000}

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
    # net_train: Train the e2e neural net
    #-----------------------------------------------------------------------------------------------
    def net_train(self, train_set, val_set=None, epochs=None, lr=None):
        """Neural net training module
        
        Inputs
        train_set: SlidingWindow object containing features x, realizations y and performance
        realizations y_perf
        val_set: SlidingWindow object containing features x, realizations y and performance
        realizations y_perf
        epochs: Number of training epochs
        lr: learning rate

        Output
        Trained model
        (Optional) val_loss: Validation loss
        """

        # Assign number of epochs and learning rate
        if epochs is None:
            epochs = self.epochs
        if lr is None:
            lr = self.lr

        # I needed to add the GPU MOD: move model to GPU if available. Better if revised for the possibility of moving to GPU all at once.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        # Build parameter groups: weight_decay on prediction weights only, zero on portfolio params
        port_param_names = {'gamma', 'delta', 'epsilon'}
        pred_params = [p for n, p in self.named_parameters() if n not in port_param_names]
        free_port_params = [p for n, p in self.named_parameters()
                            if n in ('gamma', 'delta') and p.requires_grad]
        groups = [{'params': pred_params, 'lr': lr, 'weight_decay': self.weight_decay}]
        if free_port_params:
            g_lr = self.gamma_lr if self.gamma_lr is not None else lr
            groups.append({'params': free_port_params, 'lr': g_lr, 'weight_decay': 0.0})
        if hasattr(self, 'epsilon') and self.epsilon.requires_grad:
            eps_lr = self.epsilon_lr if self.epsilon_lr is not None else lr
            groups.append({'params': [self.epsilon], 'lr': eps_lr, 'weight_decay': 0.0})
        optimizer = torch.optim.Adam(groups)

        # Number of elements in training set
        n_train = len(train_set)

        # Train the neural network
        for epoch in range(epochs):
                
            # TRAINING: forward + backward pass
            train_loss = 0
            optimizer.zero_grad() 
            for t, (x, y, y_perf) in enumerate(train_set):
                # GPU MOD: move batch to device
                x, y, y_perf = x.to(device), y.to(device), y_perf.to(device)
                
                # Forward pass: predict and optimize
                z_star, y_hat = self(x.squeeze(), y.squeeze())

                # Loss function
                if self.pred_loss is None:
                    loss = (1/n_train) * self.perf_loss(z_star, y_perf.squeeze())
                else:
                    loss = (1/n_train) * (self.perf_loss(z_star, y_perf.squeeze()) + 
                    (self.pred_loss_factor/self.n_y) * self.pred_loss(y_hat, y_perf.squeeze()[0]))

                # Backward pass: backpropagation
                loss.backward()

                # Accumulate loss of the fully trained model
                train_loss += loss.item()
        
            # Update parameters
            optimizer.step()

            # Ensure that gamma, delta, epsilon > 0 after taking a descent step
            for name, param in self.named_parameters():
                if name == 'gamma':
                    param.data.clamp_(0.0001)
                if name == 'delta':
                    param.data.clamp_(0.0001)
                if name == 'epsilon':
                    param.data.clamp_(0.0001)

            # Per-epoch rebuild of the base_rom layer with updated B
            if self.model_type == 'base_rom' and self._cov_x_cache is not None:
                B_np = self.pred_layer.weight.detach().cpu().numpy()
                sigma_mu_hat_new = B_np @ self._cov_x_cache @ B_np.T
                self.sigma_mu_hat = sigma_mu_hat_new
                self.opt_layer = base_rom(
                    self.n_y, self.n_obs, self.prisk_func,
                    sigma_mu_hat_new, self.max_weight, long_short=self.long_short
                )

        # Compute and return the validation loss of the model
        if val_set is not None:

            # Number of elements in validation set
            n_val = len(val_set)

            val_loss = 0
            with torch.no_grad():
                for t, (x, y, y_perf) in enumerate(val_set):
                    # GPU MOD: move batch to device
                    x, y, y_perf = x.to(device), y.to(device), y_perf.to(device)
                    # Predict and optimize
                    z_val, y_val = self(x.squeeze(), y.squeeze())
                
                    # Loss function
                    if self.pred_loss_factor is None:
                        loss = (1/n_val) * self.perf_loss(z_val, y_perf.squeeze())
                    else:
                        loss = (1/n_val) * (self.perf_loss(z_val, y_perf.squeeze()) + 
                        (self.pred_loss_factor/self.n_y)*self.pred_loss(y_val, y_perf.squeeze()[0]))
                    
                    # Accumulate loss
                    val_loss += loss.item()

            return val_loss

    #-----------------------------------------------------------------------------------------------
    # update_sigma_mu_hat: Update estimator covariance and rebuild base_rom layer
    #-----------------------------------------------------------------------------------------------
    def update_sigma_mu_hat(self, X_train):
        """Recompute Sigma_mu_hat = B Cov(x) B^T and rebuild the base_rom optimization layer.

        Cov(x) is cached as self._cov_x_cache so net_train() can rebuild cheaply each epoch
        without re-reading X_train. Must be called after OLS initialisation of pred_layer.
        Only meaningful for pred_model='linear' (enforced at construction via ValueError).

        Parameters
        ----------
        X_train : pd.DataFrame or torch.Tensor
            Factor data for the current training window (without a ones column).

        Returns
        -------
        dict
            Diagnostic information (updated, sigma_mu_hat_trace, rank).
        """
        import pandas as pd

        diagnostics = {}
        if self.model_type != 'base_rom':
            diagnostics['updated'] = False
            diagnostics['reason'] = f'Model type is {self.model_type}, not base_rom'
            return diagnostics

        B = self.pred_layer.weight.detach().cpu().numpy()   # (n_y, n_x)

        if isinstance(X_train, pd.DataFrame):
            cov_x = X_train.cov().values                   # (n_x, n_x)
        else:
            cov_x = torch.cov(X_train.T).cpu().numpy()

        self._cov_x_cache = cov_x                          # cached for per-epoch rebuilds
        sigma_mu_hat_new  = B @ cov_x @ B.T                # (n_y, n_y)
        self.sigma_mu_hat = sigma_mu_hat_new

        self.opt_layer = base_rom(
            self.n_y, self.n_obs, self.prisk_func,
            sigma_mu_hat_new, self.max_weight, long_short=self.long_short
        )

        eigvals = np.linalg.eigvalsh(sigma_mu_hat_new)
        rank = int(np.sum(eigvals > 1e-10 * eigvals[-1]))

        diagnostics['updated'] = True
        diagnostics['sigma_mu_hat_trace'] = float(np.trace(sigma_mu_hat_new))
        diagnostics['rank'] = rank
        return diagnostics

    #-----------------------------------------------------------------------------------------------
    # net_cv: Cross validation of the e2e neural net for hyperparameter tuning
    #-----------------------------------------------------------------------------------------------
    def net_cv(self, X, Y, lr_list, epoch_list, n_val=4):
        """Neural net cross-validation module

        Inputs
        X: Features. TrainTest object of feature timeseries data
        Y: Realizations. TrainTest object of asset time series data
        epochs: number of training passes
        lr_list: List of candidate learning rates
        epoch_list: List of candidate number of epochs
        n_val: Number of validation folds from the training dataset
        
        Output
        Trained model
        """
        results = pc.CrossVal()
        X_temp = dl.TrainTest(X.train(), X.n_obs, [1, 0])
        Y_temp = dl.TrainTest(Y.train(), Y.n_obs, [1, 0])
        for epochs in epoch_list:
            for lr in lr_list:
                
                # Train the neural network
                print('================================================')
                print(f"Training E2E {self.model_type} model: lr={lr}, epochs={epochs}")
                
                val_loss_tot = []
                for i in range(n_val-1,-1,-1):

                    # Partition training dataset into training and validation subset
                    split = [round(1-0.2*(i+1),2), 0.2]
                    X_temp.split_update(split)
                    Y_temp.split_update(split)

                    # Construct training and validation DataLoader objects
                    train_set = DataLoader(pc.SlidingWindow(X_temp.train(), Y_temp.train(), 
                                                            self.n_obs, self.perf_period))
                    val_set = DataLoader(pc.SlidingWindow(X_temp.test(), Y_temp.test(), 
                                                            self.n_obs, self.perf_period))

                    # Reset learnable parameters gamma and delta
                    self.load_state_dict(torch.load(self.init_state_path))

                    if self.pred_model == 'linear':
                        # Initialize the prediction layer weights to OLS regression weights
                        X_train, Y_train = X_temp.train(), Y_temp.train()
                        X_train.insert(0,'ones', 1.0)

                        X_train = Variable(torch.tensor(X_train.values, dtype=torch.double))
                        Y_train = Variable(torch.tensor(Y_train.values, dtype=torch.double))
                    
                        Theta = torch.inverse(X_train.T @ X_train) @ (X_train.T @ Y_train)
                        Theta = Theta.T
                        del X_train, Y_train

                        with torch.no_grad():
                            self.pred_layer.bias.copy_(Theta[:,0])
                            self.pred_layer.weight.copy_(Theta[:,1:])

                    if self.model_type == 'base_rom':
                        self.update_sigma_mu_hat(X_temp.train())

                    val_loss = self.net_train(train_set, val_set=val_set, lr=lr, epochs=epochs)
                    val_loss_tot.append(val_loss)

                    print(f"Fold: {n_val-i} / {n_val}, val_loss: {val_loss}")

                # Store results
                results.val_loss.append(np.mean(val_loss_tot))
                results.lr.append(lr)
                results.epochs.append(epochs)
                print('================================================')

        # Convert results to dataframe
        self.cv_results = results.df()
        self.cv_results.to_pickle(self.init_state_path+'_results.pkl')

        # Select and store the optimal hyperparameters
        idx = self.cv_results.val_loss.idxmin()
        self.lr = self.cv_results.lr[idx]
        self.epochs = self.cv_results.epochs[idx]

        # Print optimal parameters
        print(f"CV E2E {self.model_type} with hyperparameters: lr={self.lr}, epochs={self.epochs}")

    #-----------------------------------------------------------------------------------------------
    # net_roll_test: Test the e2e neural net
    #-----------------------------------------------------------------------------------------------
    def net_roll_test(self, X, Y, n_roll=4, lr=None, epochs=None):
        """Neural net rolling window out-of-sample test

        Inputs
        X: Features. ([n_obs+1] x n_x) torch tensor with feature timeseries data
        Y: Realizations. (n_obs x n_y) torch tensor with asset timeseries data
        n_roll: Number of training periods (i.e., number of times to retrain the model)
        lr: Learning rate for test. If 'None', the optimal learning rate is loaded
        epochs: Number of epochs for test. If 'None', the optimal # of epochs is loaded

        Output
        self.portfolio: add the backtest results to the e2e_net object
        """
        # GPU MOD: define device and move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        # Declare backtest object to hold the test results
        portfolio = pc.backtest(len(Y.test())-Y.n_obs, self.n_y, Y.test().index[Y.n_obs:])

        # Store trained gamma, delta, and epsilon values
        if self.model_type == 'nom':
            self.gamma_trained = []
        elif self.model_type == 'dro':
            self.gamma_trained = []
            self.delta_trained = []
        elif self.model_type == 'base_rom':
            self.epsilon_trained = []

        # Store the squared L2-norm of the prediction weights and their difference from OLS weights
        if self.pred_model == 'linear':
            self.theta_L2 = []
            self.theta_dist_L2 = []

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

            train_set = DataLoader(pc.SlidingWindow(X.train(), Y.train(), self.n_obs, self.perf_period))
            test_set = DataLoader(pc.SlidingWindow(X.test(), Y.test(), self.n_obs, 0))

            # Reset learnable parameters gamma and delta
            self.load_state_dict(torch.load(self.init_state_path))

            if self.pred_model == 'linear':
                # Initialize the prediction layer weights to OLS regression weights
                X_train, Y_train = X.train(), Y.train()
                X_train.insert(0,'ones', 1.0)
            
                # Move tensors to the same device as the model
                device = self.pred_layer.weight.device
                X_train = torch.tensor(X_train.values, dtype=torch.double, device=device)
                Y_train = torch.tensor(Y_train.values, dtype=torch.double, device=device)
                
                Theta = torch.inverse(X_train.T @ X_train) @ (X_train.T @ Y_train)
                Theta = Theta.T
                del X_train, Y_train

                with torch.no_grad():
                    self.pred_layer.bias.copy_(Theta[:,0])
                    self.pred_layer.weight.copy_(Theta[:,1:])

            # Update Sigma_mu_hat for base_rom using OLS-initialised B and current window's Cov(x)
            if self.model_type == 'base_rom':
                diag_info = self.update_sigma_mu_hat(X.train())
                if diag_info.get('updated', False):
                    print(f"  Sigma_mu_hat updated (trace: {diag_info['sigma_mu_hat_trace']:.2e}, rank: {diag_info['rank']})")

            train_dev = DeviceDataLoader(train_set, device)
            test_dev  = DeviceDataLoader(test_set, device)

            # Train model using all available data preceding the test window
            self.net_train(train_dev, lr=lr, epochs=epochs)

            # Store trained values of gamma, delta, and epsilon
            if self.model_type == 'nom':
                self.gamma_trained.append(self.gamma.item())
            elif self.model_type == 'dro':
                self.gamma_trained.append(self.gamma.item())
                self.delta_trained.append(self.delta.item())
            elif self.model_type == 'base_rom':
                self.epsilon_trained.append(self.epsilon.item())

            # Store the squared L2 norm of theta and distance between theta and OLS weights
            if self.pred_model == 'linear':
                theta_L2 = (torch.sum(self.pred_layer.weight**2, axis=()) + 
                            torch.sum(self.pred_layer.bias**2, axis=()))
                theta_dist_L2 = (torch.sum((self.pred_layer.weight - Theta[:,1:])**2, axis=()) + 
                                torch.sum((self.pred_layer.bias - Theta[:,0])**2, axis=()))
                self.theta_L2.append(theta_L2)
                self.theta_dist_L2.append(theta_dist_L2)

            with torch.no_grad():
                for j, (x, y, y_perf) in enumerate(test_dev):
                    # Predict and optimize
                    z_star, _ = self(x.squeeze(), y.squeeze())
                                
                    # Store portfolio weights and returns for each time step 't'
                    portfolio.weights[t] = z_star.squeeze().cpu()
                                        
                    # Perform dot product
                    portfolio.rets[t] = y_perf.squeeze().cpu() @ portfolio.weights[t]
                    t += 1


        # Reset dataset
        X, Y = X.split_update(init_split), Y.split_update(init_split)

        # Calculate the portfolio statistics using the realized portfolio returns
        portfolio.stats()

        self.portfolio = portfolio

    #-----------------------------------------------------------------------------------------------
    # load_cv_results: Load cross validation results
    #-----------------------------------------------------------------------------------------------
    def load_cv_results(self, cv_results):
        """Load cross validation results

        Inputs
        cv_results: pd.dataframe containing the cross validation results

        Outputs
        self.lr: Load the optimal learning rate
        self.epochs: Load the optimal number of epochs
        """

        # Store the cross validation results within the object
        self.cv_results = cv_results

        # Select and store the optimal hyperparameters
        idx = cv_results.val_loss.idxmin()
        self.lr = cv_results.lr[idx]
        self.epochs = cv_results.epochs[idx]

