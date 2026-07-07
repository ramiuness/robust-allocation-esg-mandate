# E2E DRO Module
#
####################################################################################################
## Import libraries
####################################################################################################
import os
import sys
import copy
import io
import tempfile
import ctypes
import contextlib
import warnings

try:
    _LIBC = ctypes.CDLL(None)            # for flushing C stdio (ECOS printf) during fd capture
except OSError:                           # non-POSIX platform: fd capture degrades gracefully
    _LIBC = None
import numpy as np
import pandas as pd
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import traceback as _traceback

import e2edro.RiskFunctions as rf
import e2edro.LossFunctions as lf
import e2edro.PortfolioClasses as pc
import e2edro.DataLoad as dl
import e2edro.observability as obs

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
# Solver configuration + robust solve  (owned by the library; no monkeypatching)
####################################################################################################
def default_solver_args(model_type):
    """Per-model_type ECOS settings reaching each problem's numerical floor, with the '*_inacc'
    band collapsed onto the strict tolerance so an unreachable solve RAISES ('optimal inaccurate'
    is otherwise silently accepted). Overridable via the solver_args constructor argument.
    (base_mod LP -> 1e-10; base_rom SOCP -> 1e-9; DRO cones -> 1e-8; see spo-critical-review.md II.)
    """
    tol = {'base_mod': 1e-10, 'base_rom': 1e-9}.get(model_type, 1e-8)
    max_iters = 10000 if model_type in ('base_mod', 'base_rom') else 20000
    return {'solve_method': 'ECOS', 'max_iters': max_iters, 'verbose': False,
            'abstol': tol, 'reltol': tol, 'feastol': tol,
            'abstol_inacc': tol, 'reltol_inacc': tol, 'feastol_inacc': tol}


@contextlib.contextmanager
def _capture_solve_output():
    """Capture the cone solver's warnings + stdout/stderr, scoped to a single solve call.

    Yields a dict filled on exit with 'warnings' (list of str), 'stdout' and 'stderr'. This is the
    reporting-era replacement for the old drop-everything `_quiet_solver`: the solver's output is
    still kept off the console (so hundreds of solves don't flood logs), but it is now *recorded*
    for robust_solve to attach to a SolveEvent instead of being silently discarded. Still scoped
    via catch_warnings, so it neither leaks to nor masks warnings raised outside the solve.
    """
    captured = {'warnings': [], 'stdout': '', 'stderr': ''}
    out, err = io.StringIO(), io.StringIO()
    with warnings.catch_warnings(record=True) as wlist, \
            contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        warnings.simplefilter('always')          # record every warning, including duplicates
        try:
            yield captured
        finally:
            captured['warnings'] = [str(w.message) for w in wlist]
            captured['stdout'], captured['stderr'] = out.getvalue(), err.getvalue()


@contextlib.contextmanager
def _capture_fd():
    """Capture C-level stdout (fd 1) into a string — e.g. ECOS's verbose iteration log, which is
    printed from C and so escapes contextlib.redirect_stdout. Yields a 1-element list filled with
    the captured text on exit. Scoped to a single solve; a temp file (not a pipe) avoids any
    buffer-full deadlock on a long log. Used only on the (rare) verbose retry.
    """
    holder = ['']
    sys.stdout.flush()
    saved = os.dup(1)
    tmp = tempfile.TemporaryFile(mode='w+b')
    os.dup2(tmp.fileno(), 1)
    try:
        yield holder
    finally:
        sys.stdout.flush()
        if _LIBC is not None:
            _LIBC.fflush(None)           # flush C stdio into tmp BEFORE restoring fd (avoids leak)
        os.dup2(saved, 1)
        os.close(saved)
        tmp.flush(); tmp.seek(0)
        holder[0] = tmp.read().decode(errors='replace')
        tmp.close()


def _emit_solve(model, event, captured, *, read_info, exc=None):
    """Build a SolveEvent from the just-completed solve and hand it to model._recorder.

    No-op when no recorder is attached. `read_info` is False on the fallback path, where
    opt_layer.info would be stale (no successful solve produced it). Context (phase/window/date)
    is read from model state, matching how _solve_log already reads _solve_phase.
    """
    rec = getattr(model, '_recorder', None)
    if rec is None:
        return
    info = (getattr(model.opt_layer, 'info', None) or {}) if read_info else {}
    tb = ''.join(_traceback.format_exception(type(exc), exc, exc.__traceback__)) if exc else None
    rec.record_solve(obs.SolveEvent(
        model=getattr(model, '_name', None),
        phase=getattr(model, '_solve_phase', None),
        window=getattr(model, '_solve_window', None),
        date=getattr(model, '_solve_date', None),
        event=event,
        solve_time=info.get('solve_time'),
        canon_time=info.get('canon_time'),
        shapes=info.get('shapes'),
        warnings=tuple(captured['warnings']) if captured else (),
        exc_type=type(exc).__name__ if exc else None,
        exc_msg=str(exc) if exc else None,
        traceback=tb,
        solver_text=(captured['stdout'] + captured['stderr']) if captured else None,
        # verbose_log = the ECOS iteration log from the verbose retry; present on both a successful
        # retry and a fallback (where that retry then failed), i.e. whenever the verbose solve ran.
        verbose_log=captured['stdout'] if (captured and event in ('retry', 'fallback')) else None,
    ))


def _grad_norm(params):
    """L2 norm of the concatenated gradients of `params` (0.0 if none carry a grad)."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    return total ** 0.5


def robust_solve(model, *params):
    """Solve model.opt_layer at model.solver_args; on failure retry at model.solve_retry_args
    (if set), then fall back to equal weight if model.solve_fallback, else re-raise. Retry/fallback
    events are recorded in model._solve_log as (model._solve_phase, 'retry'/'fallback'); the full
    per-solve record (timing, captured warnings/output, diagnostics context) is emitted to
    model._recorder when one is attached (a no-op otherwise).

    This is the owned replacement for the harness's former CvxpyLayer.forward monkeypatch: the
    solve strategy is model state, configured via the constructor, not a runtime patch of a
    third-party class. Returns the layer's output tuple (z,).
    """
    try:
        with _capture_solve_output() as cap:
            out = model.opt_layer(*params, solver_args=model.solver_args)
        _emit_solve(model, 'optimal', cap, read_info=True)
        return out
    except Exception as strict_exc:
        # Track the last failed attempt's detail so a fallback records *why* it failed (exception +
        # captured solver output), not merely that it did. `cap` from the strict solve is populated.
        last_exc, last_cap = strict_exc, cap
        if model.solve_retry_args is not None:
            try:
                # verbose=True on the retry so ECOS's iteration/residual log is captured for this
                # (rare, failing) solve; only worth it when a recorder will keep it. _capture_fd
                # grabs the C-level output (redirect_stdout alone cannot) so it never floods logs.
                retry_args = model.solve_retry_args
                if getattr(model, '_recorder', None) is not None:
                    retry_args = {**retry_args, 'verbose': True}
                with _capture_fd() as fdlog, _capture_solve_output() as cap:
                    out = model.opt_layer(*params, solver_args=retry_args)
                cap['stdout'] += fdlog[0]
                model._solve_log.append((model._solve_phase, 'retry'))
                _emit_solve(model, 'retry', cap, read_info=True, exc=strict_exc)
                return out
            except Exception as retry_exc:
                cap['stdout'] += fdlog[0]                 # the relaxed solve also failed:
                last_exc, last_cap = retry_exc, cap       # keep its exception + verbose log
        if model.solve_fallback:
            model._solve_log.append((model._solve_phase, 'fallback'))
            _emit_solve(model, 'fallback', last_cap, read_info=False, exc=last_exc)
            z = torch.full((model.n_y, 1), 1.0 / model.n_y, dtype=torch.double,
                           device=model.pred_layer.weight.device, requires_grad=True)
            return (z,)
        raise


####################################################################################################
# Shared OLS warm-start
####################################################################################################
def ols_theta(X_df, Y_df):
    """OLS with intercept. Returns Theta (n_y x [1+n_x]) = [bias | weights] as a double tensor.

    Single source of truth for the OLS warm-start that was previously duplicated across net_cv,
    net_roll_test, BaseModels.pred_then_opt and gamma_range, and the harness. 'ones' is the FIRST
    column so Theta[:, 0] is the bias; solved with torch.linalg.lstsq in double.
    """
    Xt = X_df.copy()
    Xt.insert(0, 'ones', 1.0)
    X = torch.tensor(Xt.values, dtype=torch.double)
    Y = torch.tensor(Y_df.values, dtype=torch.double)
    return torch.linalg.lstsq(X, Y).solution.T


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
                pred_model='linear', pred_loss_factor=0.5, perf_period=13, train_pred=True, train_gamma=True, train_delta=True, train_epsilon=True, set_seed=42, epochs=10, lr=1e-3, epsilon_lr=None, weight_decay=0.0, gamma_lr=None, long_short=False, cache_path='./cache/', max_weight=1.0, solver_args=None, solve_retry_args=None, solve_fallback=False):
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

        # Local RNG for parameter init. Draw gamma/delta/epsilon from a private Generator so the
        # draw depends only on this model's seed -- not on global RNG state or construction order
        # -- and never perturbs the global stream. (pred_layer's random init is intentionally
        # left to the global default: it is always OLS-overwritten before use.)
        self.seed = set_seed
        gen = torch.Generator()
        if set_seed is not None:
            gen.manual_seed(set_seed)

        self.n_x = n_x
        self.n_y = n_y
        self.n_obs = n_obs
        self.max_weight = max_weight  # Max weight per asset for diversification

        # Feasibility: an active per-asset cap must admit a feasible budget (sum(z)==1).
        # Checked here for every opt_layer (previously only base_rom validated it).
        if max_weight < 1.0 and max_weight * n_y < 1.0:
            raise ValueError(
                f"Infeasible: max_weight={max_weight} with n_y={n_y} assets. "
                f"Need max_weight >= {1.0/n_y:.4f} (= 1/n_y) for feasibility."
            )
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
        self.gamma = nn.Parameter(torch.empty(1).uniform_(0.02, 0.1, generator=gen))
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
            self.epsilon = nn.Parameter(torch.empty(1).uniform_(0.1, 1.0, generator=gen))
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
            self.delta = nn.Parameter(torch.empty(1).uniform_(lb, ub, generator=gen))
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

        # LAYER: Optimization model. base_rom's Sigma_mu_hat = B Cov(x) B^T needs data (Cov(x))
        # and the OLS-fitted B, neither available at construction -- so its opt_layer is built
        # lazily by fit_predictor() and stays None until then (forward() raises if used unfitted).
        # No identity placeholder: there is no correct placeholder value, so we forbid use instead.
        if opt_layer == 'base_rom':
            self.sigma_mu_hat = None
            self._cov_x_cache = None
            self.opt_layer = None
        else:
            self.opt_layer = eval(opt_layer)(n_y, n_obs, self.prisk_func,
                                             max_weight=max_weight, long_short=long_short)
        # Store reference path to store model data
        self.cache_path = cache_path

        # Solver strategy (owned; no monkeypatch). solver_args defaults per model_type with the
        # '*_inacc' band collapsed so an inaccurate solve raises. Optional retry/equal-weight
        # fallback are off by default (fail-loud); the experiment harness turns them on.
        self.solver_args = solver_args if solver_args is not None else default_solver_args(self.model_type)
        self.solve_retry_args = solve_retry_args
        self.solve_fallback = solve_fallback
        self._solve_log = []       # accumulates (phase, 'retry'/'fallback') across windows
        self._solve_phase = None   # set by the caller (calibrate/train/infer) before solving
        self._solve_window = None  # roll-window index; set by net_roll_test / infer_window
        self._solve_date = None    # decision date; set per step at inference
        self._recorder = None      # optional obs.Recorder sink; None => zero-overhead no-op

        # Cast all parameters/submodules to double. MUST be after every parameter and
        # submodule exists (gamma/delta/epsilon, pred_layer, opt_layer); calling it at the
        # top of __init__ would be a no-op while SlidingWindow always emits float64.
        self.double()

        # In-memory pristine snapshot of the just-initialised learnable state. Each roll / CV fold
        # resets from this (see net_roll_test / net_cv). Held per-object, so two same-config models
        # (e.g. tv and hellinger, both model_type='dro') can never clobber each other's init -- the
        # previous model_type-keyed on-disk checkpoint did, silently loading the wrong delta. The
        # opt_layer (CvxpyLayer) contributes no state_dict keys, so this is unaffected by base_rom's
        # placeholder/rebuild.
        self.opt_layer_name = opt_layer
        self._init_state = copy.deepcopy(self.state_dict())

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

        # Optimize z per scenario via the owned robust-solve strategy (self.solver_args, with
        # optional retry/fallback). Determine whether nominal, dro, base_mod or base_rom model.
        if self.model_type == 'nom':
            z_star, = robust_solve(self, ep, y_hat, self.gamma)
        elif self.model_type == 'dro':
            z_star, = robust_solve(self, ep, y_hat, self.gamma, self.delta)
        elif self.model_type == 'base_mod':
            z_star, = robust_solve(self, y_hat)
        elif self.model_type == 'base_rom':
            if self.opt_layer is None:
                raise RuntimeError(
                    "base_rom model is not fitted: call fit_predictor(X_train, Y_train) to build "
                    "Sigma_mu_hat = B Cov(x) B^T before forward()/net_train()."
                )
            z_star, = robust_solve(self, y_hat, self.epsilon)

        return z_star, y_hat

    def _emit_epoch(self, epoch, loss_total, loss_task, loss_pred, grad_norm_pred, grad_norm_robust):
        """Emit one EpochRecord of learning telemetry to the recorder. No-op without one.

        loss_val is left None here (net_train computes validation loss once, after all epochs);
        theta_dist_l2 is measured against the OLS warm-start stashed by fit_predictor.
        """
        rec = self._recorder
        if rec is None:
            return
        w, b = self.pred_layer.weight.detach(), self.pred_layer.bias.detach()
        theta_l2 = float((w ** 2).sum() + (b ** 2).sum())
        theta_dist = None
        theta_ols = getattr(self, '_ols_theta', None)
        if theta_ols is not None:
            theta_dist = float(((w - theta_ols[:, 1:]) ** 2).sum() + ((b - theta_ols[:, 0]) ** 2).sum())

        def _param(name):
            p = getattr(self, name, None)
            return float(p.item()) if p is not None else None

        rec.record_epoch(obs.EpochRecord(
            model=getattr(self, '_name', None), window=getattr(self, '_solve_window', None),
            epoch=epoch, loss_total=loss_total, loss_task=loss_task, loss_pred=loss_pred,
            gamma=_param('gamma'), delta=_param('delta'), epsilon=_param('epsilon'),
            grad_norm_pred=grad_norm_pred, grad_norm_robust=grad_norm_robust,
            decay_norm=float(self.weight_decay) * (theta_l2 ** 0.5),
            theta_l2=theta_l2, theta_dist_l2=theta_dist))

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

        # base_rom must be fitted first: Sigma_mu_hat needs Cov(x) + the OLS-fitted B. Fail loud
        # rather than silently training epsilon against a stale/absent geometry.
        if self.model_type == 'base_rom' and self._cov_x_cache is None:
            raise RuntimeError(
                "base_rom is not fitted: call fit_predictor(X_train, Y_train) before net_train()."
            )

        # Build parameter groups: weight_decay on prediction weights only, zero on portfolio params
        port_param_names = {'gamma', 'delta', 'epsilon'}
        pred_params = [p for n, p in self.named_parameters() if n not in port_param_names]
        free_port_params = [p for n, p in self.named_parameters()
                            if n in ('gamma', 'delta') and p.requires_grad]
        groups = [{'params': pred_params, 'lr': lr, 'weight_decay': self.weight_decay}]
        robust_params = []                       # gamma/delta/epsilon, for per-epoch grad-norm report
        if free_port_params:
            g_lr = self.gamma_lr if self.gamma_lr is not None else lr
            groups.append({'params': free_port_params, 'lr': g_lr, 'weight_decay': 0.0})
            robust_params += free_port_params
        if hasattr(self, 'epsilon') and self.epsilon.requires_grad:
            eps_lr = self.epsilon_lr if self.epsilon_lr is not None else lr
            groups.append({'params': [self.epsilon], 'lr': eps_lr, 'weight_decay': 0.0})
            robust_params.append(self.epsilon)
        optimizer = torch.optim.Adam(groups)

        # Number of elements in training set
        n_train = len(train_set)

        # Train the neural network
        rec = self._recorder                     # per-epoch telemetry sink (None => no-op)
        for epoch in range(epochs):

            # TRAINING: forward + backward pass
            train_loss = 0.0
            task_sum, pred_sum = 0.0, 0.0
            optimizer.zero_grad()
            for t, (x, y, y_perf) in enumerate(train_set):
                # GPU MOD: move batch to device
                x, y, y_perf = x.to(device), y.to(device), y_perf.to(device)

                # Forward pass: predict and optimize
                z_star, y_hat = self(x.squeeze(), y.squeeze())

                # Loss = task loss + optional prediction co-objective. Split into named terms so the
                # per-epoch report can attribute the two; the combined `loss` graph is identical to
                # the former inline expression, so training is numerically unchanged.
                task = self.perf_loss(z_star, y_perf.squeeze())
                if self.pred_loss is None:
                    loss = (1/n_train) * task
                    pred = None
                else:
                    pred = self.pred_loss(y_hat, y_perf.squeeze()[0])
                    loss = (1/n_train) * (task + (self.pred_loss_factor/self.n_y) * pred)

                # Backward pass: backpropagation
                loss.backward()

                # Accumulate loss of the fully trained model
                train_loss += loss.item()
                if rec is not None:
                    task_sum += task.item() / n_train
                    if pred is not None:
                        pred_sum += (self.pred_loss_factor/self.n_y) * pred.item() / n_train

            # Gradient norms must be read after backward, before optimizer.step() updates the params.
            grad_norm_pred = grad_norm_robust = None
            if rec is not None:
                grad_norm_pred = _grad_norm(pred_params)
                grad_norm_robust = _grad_norm(robust_params)

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

            # Per-epoch rebuild of the base_rom layer with the updated B -- only when B is actually
            # learning (train_pred=True). When B is frozen the layer built at fit_predictor() is
            # already correct, so rebuilding every epoch would be byte-identical wasted work.
            if self.model_type == 'base_rom' and self.pred_layer.weight.requires_grad:
                self._rebuild_opt_layer()

            # Per-epoch learning telemetry (no-op when no recorder is attached)
            if rec is not None:
                self._emit_epoch(epoch, train_loss, task_sum, pred_sum,
                                 grad_norm_pred, grad_norm_robust)

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
                    if self.pred_loss is None:
                        loss = (1/n_val) * self.perf_loss(z_val, y_perf.squeeze())
                    else:
                        loss = (1/n_val) * (self.perf_loss(z_val, y_perf.squeeze()) + 
                        (self.pred_loss_factor/self.n_y)*self.pred_loss(y_val, y_perf.squeeze()[0]))
                    
                    # Accumulate loss
                    val_loss += loss.item()

            return val_loss

    #-----------------------------------------------------------------------------------------------
    # fit_predictor / _rebuild_opt_layer: OLS warm-start and (base_rom) estimator-covariance build
    #-----------------------------------------------------------------------------------------------
    def _rebuild_opt_layer(self):
        """(Re)build base_rom's SOCP layer from the current B and the cached Cov(x).

        One place for both the initial build (fit_predictor) and the per-epoch rebuild (net_train),
        so the estimation-robust geometry is always Sigma_mu_hat = B Cov(x) B^T for the live B.
        """
        B = self.pred_layer.weight.detach().cpu().numpy()          # (n_y, n_x)
        self.sigma_mu_hat = B @ self._cov_x_cache @ B.T            # (n_y, n_y)
        self.opt_layer = base_rom(
            self.n_y, self.n_obs, self.prisk_func,
            self.sigma_mu_hat, self.max_weight, long_short=self.long_short
        )

    def fit_predictor(self, X_train, Y_train):
        """OLS-warm-start the prediction layer from one training window; for base_rom also cache
        Cov(x) and (re)build the estimation-robust opt_layer. Single entry point replacing the old
        external 'OLS init + update_sigma_mu_hat' pair (which forced a public method + placeholder).

        X_train : feature DataFrame (standardized, no ones column) or tensor.
        Y_train : return DataFrame/tensor, positionally aligned with X_train.
        Returns Theta (n_y x [1+n_x]) on the pred_layer's device.
        """
        Theta = ols_theta(X_train, Y_train)
        device = self.pred_layer.weight.device
        Theta = Theta.to(device)
        with torch.no_grad():
            self.pred_layer.bias.copy_(Theta[:, 0])
            self.pred_layer.weight.copy_(Theta[:, 1:])
        if self.model_type == 'base_rom':
            if isinstance(X_train, pd.DataFrame):
                self._cov_x_cache = X_train.cov().values
            else:
                self._cov_x_cache = torch.cov(X_train.T).cpu().numpy()
            self._rebuild_opt_layer()
        self._ols_theta = Theta          # OLS warm-start reference for per-epoch theta_dist_l2
        return Theta

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

                    # Re-fit feature standardization on this fold's train window, apply to its val window.
                    mu = X_temp.train().mean()
                    sigma = X_temp.train().std().replace(0.0, 1.0)
                    Xtr, Xval = (X_temp.train() - mu) / sigma, (X_temp.test() - mu) / sigma

                    # Construct training and validation DataLoader objects
                    train_set = DataLoader(pc.SlidingWindow(Xtr, Y_temp.train(),
                                                            self.n_obs, self.perf_period))
                    val_set = DataLoader(pc.SlidingWindow(Xval, Y_temp.test(),
                                                            self.n_obs, self.perf_period))

                    # Reset learnable parameters gamma and delta
                    self.load_state_dict(self._init_state)

                    # OLS warm-start + (base_rom) Sigma_mu_hat build, in one call
                    if self.pred_model == 'linear':
                        self.fit_predictor(Xtr, Y_temp.train())

                    val_loss = self.net_train(train_set, val_set=val_set, lr=lr, epochs=epochs)
                    val_loss_tot.append(val_loss)

                    print(f"Fold: {n_val-i} / {n_val}, val_loss: {val_loss}")

                # Store results
                results.val_loss.append(np.mean(val_loss_tot))
                results.lr.append(lr)
                results.epochs.append(epochs)
                print('================================================')

        # Convert results to dataframe. Disambiguate the pickle by opt_layer + pred_model + seed
        # so tv/hellinger (both model_type='dro') and different-seed runs never collide.
        self.cv_results = results.df()
        cv_path = f"{self.cache_path}cv_{self.opt_layer_name}_{self.pred_model}_seed{self.seed}.pkl"
        os.makedirs(os.path.dirname(cv_path) or '.', exist_ok=True)
        self.cv_results.to_pickle(cv_path)

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
            self._solve_window = i          # tag every solve in this window (report context)

            split[0] = init_split[0] + win_size * i
            if i < n_roll-1:
                split[1] = win_size
            else:
                split[1] = 1 - split[0]

            X.split_update(split), Y.split_update(split)

            # Re-fit feature standardization on this roll's train window, apply to its test window.
            mu = X.train().mean()
            sigma = X.train().std().replace(0.0, 1.0)
            Xtr, Xte = (X.train() - mu) / sigma, (X.test() - mu) / sigma

            train_set = DataLoader(pc.SlidingWindow(Xtr, Y.train(), self.n_obs, self.perf_period))
            test_set = DataLoader(pc.SlidingWindow(Xte, Y.test(), self.n_obs, 1))

            # Reset learnable parameters gamma and delta
            self.load_state_dict(self._init_state)

            # OLS warm-start + (base_rom) Sigma_mu_hat build, in one call. Theta is kept for the
            # theta_dist_L2 bookkeeping below.
            Theta = None
            if self.pred_model == 'linear':
                Theta = self.fit_predictor(Xtr, Y.train())

            train_dev = DeviceDataLoader(train_set, device)
            test_dev  = DeviceDataLoader(test_set, device)

            # Train model using all available data preceding the test window
            self._solve_phase = 'train'
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

            self._solve_phase = 'infer'
            test_dates = Y.test().index[self.n_obs:]
            with torch.no_grad():
                for j, (x, y, y_perf) in enumerate(test_dev):
                    # Predict and optimize
                    self._solve_date = test_dates[j] if j < len(test_dates) else None
                    z_star, _ = self(x.squeeze(), y.squeeze())

                    # Store portfolio weights and returns for each time step 't'
                    portfolio.weights[t] = z_star.squeeze().cpu()

                    # Perform dot product
                    portfolio.rets[t] = y_perf.squeeze().cpu() @ portfolio.weights[t]
                    t += 1

            # Window boundary: let the recorder compute lazy diagnostics for this window (it only
            # does work if the window saw a retry/fallback). Passes the standardized train slice.
            if self._recorder is not None:
                self._recorder.on_window(self, i, test_dates, Xtr, Y.train())


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

