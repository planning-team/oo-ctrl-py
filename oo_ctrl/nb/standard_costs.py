import numba.typed
import numpy as np

from typing import Tuple, Callable, Union, Optional, Dict, Any
from numba import njit, prange
from numba.typed import Dict as NumbaDict
from oo_ctrl.nb.core import AbstractNumbaCost
from oo_ctrl.nb.util import wrap_angle, extract_dims


class EuclideanGoalCost(AbstractNumbaCost):
    r"""Calculates (weighted) Euclidean distance based cost.
    
    Given the current state :math:`x` and the goal state :math:`g`, the cost is calculated:
    :math:`(x - g)^T*Q*(x-g)` if ``squared`` is ``True``, else :math:`\sqrt{(x - g)^T*Q*(x-g)}`,
    where :math:`Q` is a diagonal matrix with positive elements.
    
    The state variable :math:`x` is extracted from ``state`` argument using ``state_dims`` parameter.
    If ``state_dims`` is integer, it defines a slice ``x = state[..., :state_dims]``.
    If ``state_dims`` is a tuple, it defines dim indices to select a slice: ``x = state[..., state_dims]``.
    If ``state_dims`` is None, :math:`x` is taken as is: ``x = state``.
    
    Matrix :math:`Q` is defined via the ``q`` parameter.
    If ``q`` is an float, the corresponding matrix diagonal will have same elements.
    If ``q`` is a tuple, it defines a diagonal for the matrix.
    
    The goal `math`:`g` is extracted from the ``observation`` using the key defined in ``goal_key`` parameter.
    Goal is taken from the ``observation`` as is: ``g = observation[goal_key]``.

    Args:
        Q_diag (Union[float, Tuple[float, ...]]): Value (or values) for the :math:`Q` diagonal matrix.
        squared (bool): If False, the squared root is applied to the quadratic form.
        state_dims (Optional[Union[int, Tuple[int, ...]]], optional): Dimensions to extract :math:`x` from. Defaults to None.
        goal_key (str, optional): Key to extract the goal :math:`g` from. Defaults to "goal".
    """

    def __init__(self,
                 Q_diag: Tuple[float, ...],
                 squared: bool,
                 state_dims: Optional[Union[int, Tuple[int, ...]]] = None,
                 goal_key: str = "goal",
                 name: str = "euclidean_goal"):
        super(EuclideanGoalCost, self).__init__(name=name)
        if isinstance(Q_diag, float) or isinstance(Q_diag, int):
            assert Q_diag > 0., f"Q must be > 0 if single value, got {Q_diag}"
            if state_dims is not None:
                if isinstance(state_dims, int):
                    Q_diag = np.eye(state_dims) * Q_diag
                else:
                    Q_diag = np.eye(len(state_dims)) * Q_diag
        else:
            if state_dims is not None:
                q_len = len(Q_diag)
                if isinstance(state_dims, int):
                    state_dims_len = state_dims
                else:
                    state_dims_len = len(state_dims)
                assert q_len == state_dims_len, f"Length of Q_diag must match state_dims \
                    final size, got {q_len} and {state_dims_len}"
            Q_diag = np.diag(Q_diag) 
            assert (np.diag(Q_diag) > 0.).all(), f"All elements of Q_diag must be > 0"
            
        self._Q = Q_diag
        self._squared = squared
        self._state_dims = state_dims
        self._goal_key = goal_key

    def make(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        Q = self._Q
        squared = self._squared
        state_dims = self._state_dims
        goal_key = self._goal_key

        @njit(parallel=True)
        def _fn(state: np.ndarray, 
                control: np.ndarray, 
                observation: NumbaDict[str, np.ndarray]) -> np.ndarray:
            x = extract_dims(state, state_dims)
            g = observation[goal_key]
            
            n_samples = x.shape[0]
            horizon = x.shape[1]
            state_dim = x.shape[2]
            result = np.zeros((n_samples, horizon), dtype=state.dtype)
            
            for i in prange(n_samples):
                for j in prange(horizon):
                    dist = np.zeros(1)
                    for k in prange(state_dim):
                        dist = dist + Q[k, k] * (x[i, j, k] - g[k]) ** 2
                    if not squared:
                        dist = np.sqrt(dist)
                    result[i, j] = dist[0]

            return result
        
        return _fn
