import numpy as np

from typing import List, Tuple, Union, Dict, Optional, Any
from oo_ctrl.np.core import (AbstractNumPyMPC,
                             AbstractNumPyModel,
                             AbstractNumPyCost,
                             AbstractActionSampler)
from oo_ctrl.np.util import vec_mat_vec


class MPPI(AbstractNumPyMPC):
    
    def __init__(self,
                 horizon: int,
                 n_samples: int,
                 lmbda: float,
                 model: AbstractNumPyModel,
                 cost: Union[List[Union[Tuple[float, AbstractNumPyCost], AbstractNumPyCost]], 
                             Union[Tuple[float, AbstractNumPyCost], AbstractNumPyCost]],
                 sampler: AbstractActionSampler,
                 biased: bool = False
                 ):
        assert isinstance(horizon, int) and horizon > 0, f"horizon must be int > 0, got {horizon}"
        assert isinstance(n_samples, int) and n_samples > 0, f"n_samples must be int > 0, got {n_samples}"
        if not biased:
            assert hasattr(sampler, "covariance_matrix"), f"For unbiased version of MPPI, sampler must have 'covariance_matrix' attribute"
        
        composite_cost = []
        if isinstance(cost, AbstractNumPyCost):
            composite_cost.append((1., cost))
        else:
            for cost_component in cost:
                if isinstance(cost_component, AbstractNumPyCost):
                    composite_cost.append((1., cost_component))
                else:
                    assert (len(cost_component) == 2) \
                        and(isinstance(cost_component[0], int) \
                        or isinstance(cost_component[0], float)) \
                        and cost_component[0] > 0 \
                        and isinstance(cost_component, AbstractNumPyCost), \
                            f"If tuple, cost component must have format (weight, cost) and weight > 0"
                    composite_cost.append(cost_component)
        cost = composite_cost
            
        super(MPPI, self).__init__()
        self._horizon = horizon
        self._n_samples = n_samples
        self._lambda = lmbda
        self._model = model
        self._cost = cost
        self._sampler = sampler
        self._biased = biased
        
        self._u_prev = np.zeros((horizon, model.control_lb.shape[0]))
        
    def step(self,
             current_state: np.ndarray,
             observation: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        epsilon = self._sampler(n_samples=self._n_samples,
                                horizon=self._horizon,
                                observation=observation) # (n_samples, horizon, dim)
        u_eps = np.tile(self._u_prev, (self._n_samples, 1, 1)) + epsilon # (n_samples, horizon, dim)
        
        x_prev = np.tile(current_state, (self._n_samples, 1))
        x_seq = [x_prev]
        for i in range(self._horizon):
            x_prev = self._model(x_prev, u_eps[:, i, :])
            x_seq.append(x_prev)
        x_seq = np.stack(x_seq, axis=1)
        
        s = self._calculate_costs(x_seq, u_eps, observation) # (n_samples,)
        
        if not self._biased:
            cov_inv = np.linalg.inv(self._sampler.covariance_matrix)
            s = s + (self._lambda / 2.) * vec_mat_vec(u_eps, cov_inv, u_eps).sum(axis=1)
            s = s + self._lambda * vec_mat_vec(u_eps, cov_inv, epsilon).sum(axis=1)
        
        beta = np.min(s)
        
        s_exp = np.exp(-(1. / self._lambda) * (s - beta))
        eta = np.sum(s_exp)
        w = s_exp / eta # (n_samples,)
        
        delta_u = np.tensordot(w, epsilon, axes=(0, 0))
        u = self._u_prev + delta_u
        u = self._model.clip(u)
        
        self._u_prev[:-1] = u[1:, :].copy()
        self._u_prev[-1] = u[-1].copy()
        
        info = {
            "u_seq": u.copy()
        }
        
        return u[0], info
        
        
    def _calculate_costs(self,
                         x: np.ndarray,
                         u: np.ndarray,
                         observation: Optional[Dict[str, Any]]) -> np.ndarray:
        result = 0.
        for w, cost in self._cost:
            cost_values_horizon = cost(x, u, observation)
            cost_sum = np.sum(cost_values_horizon, axis=1)
            result = result + w * cost_sum
        return result
