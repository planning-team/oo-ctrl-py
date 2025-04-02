import numpy as np

from typing import List, Tuple, Union, Dict, Optional, Any
from numba import njit, prange
from numba.typed import Dict as NumbaDict
from oo_ctrl.nb.core import (AbstractNumbaModel,
                             AbstractNumbaCost)
from oo_ctrl.nb.standard_costs import CompositeCost


@njit(parallel=True)
def _vec_mat_vec(u: np.ndarray,
                 sigma_inv: np.ndarray,
                 eps: np.ndarray) -> np.ndarray:
    n_samples = u.shape[0]
    horizon = u.shape[1]
    result = np.zeros((n_samples, horizon))
    for i in prange(n_samples):
        for j in prange(horizon):
            result[i, j] = u[i, j].T @ sigma_inv @ eps[i, j]
    return result


class MPPI:
    
    def __init__(self,
                 horizon: int,
                 n_samples: int,
                 stds: Tuple[float, ...],
                 lmbda: float,
                 model: AbstractNumbaModel,
                 cost: Union[AbstractNumbaCost, List[AbstractNumbaCost]]):
        assert isinstance(horizon, int) and horizon > 0, f"horizon must be int > 0, got {horizon}"
        assert isinstance(n_samples, int) and n_samples > 0, f"n_samples must be int > 0, got {n_samples}"

        if isinstance(cost, list):
            cost = CompositeCost(cost)
        elif not isinstance(cost, AbstractNumbaCost):
            raise ValueError(f"cost must be AbstractNumbaCost or list of AbstractNumbaCost, got {type(cost)}")
            
        # super(MPPI, self).__init__()
        self._horizon = horizon
        self._n_samples = n_samples
        self._stds = np.array(stds)
        self._sigma = np.diag(stds) ** 2
        self._sigma_inv = np.linalg.inv(self._sigma)
        self._mu = np.zeros(self._sigma.shape[0])
        self._lambda = lmbda
        self._model = model
        self._cost = cost

        self._control_fn = self._make_controller_fn()
        
        self._u_prev = np.zeros((horizon, model.control_lb.shape[0]))

    def step(self,
             current_state: np.ndarray,
             observation: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        observation_nb = NumbaDict()
        for k, v in observation.items():
            observation_nb[k] = v

        u_new = self._control_fn(current_state, self._u_prev, observation_nb)
        
        u_new = np.clip(u_new, self._model.control_lb, self._model.control_ub)   
        self._u_prev[:-1] = u_new[1:, :]
        self._u_prev[-1] = u_new[-1]

        return u_new[0], {"u_seq": u_new}

    def _make_controller_fn(self):
        model_fn = self._model.make()
        cost_fn = self._cost.make()
        mu = self._mu
        stds = self._stds
        sigma = self._sigma
        sigma_inv = self._sigma_inv
        lambda_ = self._lambda
        horizon = self._horizon
        n_samples = self._n_samples
        u_dim = self._model.control_lb.shape[0]
        
        @njit(parallel=True)
        def _fn(current_state: np.ndarray,
                u_prev: np.ndarray,
                observation: NumbaDict[str, np.ndarray]) -> np.ndarray:
            x_dim = current_state.shape[0]
            
            u_eps = np.zeros((n_samples, horizon, u_dim))
            epsilon = np.zeros((n_samples, horizon, u_dim))
            for i in prange(n_samples):
                for j in prange(horizon):
                    for k in prange(u_dim):
                        epsilon[i, j, k] = np.random.normal(0., stds[k])
                    u_eps[i, j, :] = u_prev[j, :] + epsilon[i, j, :]
            
            x_seq = np.zeros((n_samples, horizon + 1, x_dim))
            x_seq[:, 0, :] = current_state
            for j in prange(horizon):
                x_new = model_fn(x_seq[:, j, :], u_eps[:, j, :])
                x_seq[:, j + 1, :] = x_new
            
            s = cost_fn(x_seq, u_eps, observation) # (n_samples, horizon)
            s = np.sum(s, axis=1)  # n_samples
            s = s + lambda_ * _vec_mat_vec(u_eps, sigma_inv, epsilon).sum(axis=1)
            
            beta = np.min(s)
            
            s_exp = np.exp(-(1. / lambda_) * (s - beta))
            eta = np.sum(s_exp)
            w = s_exp / eta  # (n_samples,)
            
            delta_u = np.zeros((horizon, u_dim))
            for j in prange(horizon):
                u_step = np.zeros((u_dim,))
                for i in prange(n_samples):
                    u_step = u_step + w[i] * epsilon[i, j, :]
                delta_u[j, :] = u_step
            u = u_prev + delta_u

            return u
        
        return _fn
