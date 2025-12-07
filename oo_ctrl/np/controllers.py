import numpy as np

from typing import List, Tuple, Union, Dict, Optional, Any
from oo_ctrl.np.core import (AbstractNumPyMPC,
                             AbstractNumPyModel,
                             AbstractNumPyCost,
                             AbstractActionSampler,
                             AbstractStateTransform,
                             AbstractPresampler)
from oo_ctrl.np.cost_monitor import CostMonitor
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
                 biased: bool = False,
                 state_transform: Optional[AbstractStateTransform] = None,
                 presampler: Optional[AbstractPresampler] = None,
                 cost_monitor: bool = False,
                 return_state_seq: bool = False,
                 return_samples: bool = False,
                 return_pre_samples: bool = False
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
        self._state_transform = state_transform
        self._presampler = presampler
        
        self._u_prev = np.zeros((horizon, model.control_lb.shape[0]))
        
        self._cost_monitor = CostMonitor() if cost_monitor else None
        self._return_state_seq = return_state_seq
        self._return_samples = return_samples
        self._return_pre_samples = return_pre_samples
        
    @property
    def cost_monitor(self) -> Optional[CostMonitor]:
        return self._cost_monitor
        
    def step(self,
             current_state: np.ndarray,
             observation: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        info = {}
        # Transform state to the dynamics model space if needed
        if self._state_transform is not None:
            current_state = self._state_transform.inverse(current_state)

        # Inint nominal trajectory
        u_nominal, x_seq_pre_samples, x_seq_pre_samples_min = self._init_nominal(current_state, observation)
        if self._return_pre_samples and x_seq_pre_samples is not None:
            info["x_seq_pre_samples"] = x_seq_pre_samples
            info["x_seq_pre_samples_min"] = x_seq_pre_samples_min
        u_nominal = np.tile(u_nominal, (self._n_samples, 1, 1)) # (n_samples, horizon, dim)
        
        # Perturbation samples
        epsilon = self._sampler(n_samples=self._n_samples,
                                horizon=self._horizon,
                                observation=observation) # (n_samples, horizon, dim)

        # Notation from the paper: v = u + eps
        v_seq = u_nominal + epsilon # (n_samples, horizon, dim)
        
        # Do rollout with perturbed control sequences
        x_prev = np.tile(current_state, (self._n_samples, 1))
        x_seq = []
        for i in range(self._horizon):
            x_prev = self._model(x_prev, self._model.clip(v_seq[:, i, :]))
            x_seq.append(x_prev)
        x_seq = np.stack(x_seq, axis=1) # (n_samples, horizon, dim)
        # Transform states for cost calculation if needed
        if self._state_transform is not None:
            x_seq = self._state_transform.forward(x_seq)
        if self._return_samples:
            info["x_seq_samples"] = x_seq.copy()
        
        # Calculate costs of the perturbed trajectories
        # Final stage costs must be calculated inside the cost functions implementations
        s, _ = self._calculate_costs(x_seq, v_seq, observation) # (n_samples,)
        
        if not self._biased:
            cov_inv = np.linalg.inv(self._sampler.covariance_matrix)
            # In paper, this cost addition is quad-form of nominal control sequence,
            # inverse covariance matrix and perturbations
            # We account them by summing over horizon
            s = s + self._lambda * vec_mat_vec(u_nominal, cov_inv, epsilon).sum(axis=1)
        
        beta = np.min(s)
        
        s_exp = np.exp(-(1. / self._lambda) * (s - beta))
        eta = np.sum(s_exp)
        w = s_exp / eta # (n_samples,)
        
        delta_u = np.tensordot(w, epsilon, axes=(0, 0))
        u = self._u_prev + delta_u
        u = self._model.clip(u)
        
        self._u_prev[:-1] = u[1:, :].copy()
        self._u_prev[-1] = u[-1].copy()
    
        info["u_seq"] = u.copy()

        if self._return_state_seq or self._cost_monitor:
            x_prev = current_state.copy()
            x_seq = []
            for i in range(self._horizon):
                x_prev = self._model(x_prev, u[i, :])
                x_seq.append(x_prev)
            x_seq = np.stack(x_seq, axis=0)
            if self._state_transform is not None:
                x_seq = self._state_transform.forward(x_seq)

            info["x_seq"] = x_seq
            if self._cost_monitor:
                self._log_result_cost(x_seq, u, observation)        
        
        return u[0], info
        
    def _calculate_costs(self,
                         x: np.ndarray,
                         u: np.ndarray,
                         observation: Optional[Dict[str, Any]]) -> Tuple[np.ndarray,
                                                                         Dict[str, np.ndarray]]:
        result = 0.
        values_horizon = {}
        for w, cost in self._cost:
            cost_values_horizon = cost(x, u, observation)
            values_horizon[cost.name] = cost_values_horizon
            cost_sum = np.sum(cost_values_horizon, axis=1)
            result = result + w * cost_sum
        return result, values_horizon

    def _log_result_cost(self,
                         x_seq: np.ndarray,
                         u_seq: np.ndarray,
                         observation: Optional[Dict[str, Any]]):
        x_seq = x_seq[np.newaxis, ...]
        u_seq = u_seq[np.newaxis, ...]
        _, values_horizon = self._calculate_costs(x_seq, 
                                                  u_seq,
                                                  observation)
        for k, v in values_horizon.items():
            self._cost_monitor.log_cost(k, v[0, :])

    def _init_nominal(self,
                      current_state: np.ndarray,
                      observation: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, 
                                                                             Optional[np.ndarray],
                                                                             Optional[np.ndarray]]:
        # No presampler - initi with previous solution
        if self._presampler is None:
            return self._u_prev, None, None

        # Sample candidates
        u_seq = self._presampler.sample(state=current_state, 
                                        observation=observation) # (n_samples, horizon, dim)
        # Add previous solution to the candidates
        u_seq = np.concat((u_seq, self._u_prev[np.newaxis, :, :]), axis=0)
        u_seq = self._model.clip(u_seq)

        # Do rollout
        x_prev = np.tile(current_state, (u_seq.shape[0], 1))
        x_seq = []
        for i in range(self._horizon):
            x_prev = self._model(x_prev, u_seq[:, i, :])
            x_seq.append(x_prev)
        x_seq = np.stack(x_seq, axis=1) # (n_samples, horizon, dim)

        s, _ = self._calculate_costs(x_seq, u_seq, observation) # (n_samples,)
        min_idx = np.argmin(s)
        u_nominal = u_seq[min_idx]

        return u_nominal, x_seq, x_seq[min_idx]
