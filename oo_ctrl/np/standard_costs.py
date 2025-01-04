import numpy as np

from typing import Union, Dict, Any, Tuple, Optional
from oo_ctrl.np.core import AbstractNumPyCost
from oo_ctrl.np.util import extract_dims


class EuclideanCost(AbstractNumPyCost):
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
                 Q_diag: Union[float, Tuple[float, ...]],
                 squared: bool,
                 state_dims: Optional[Union[int, Tuple[int, ...]]] = None,
                 goal_key: str = "goal",
                 name: str = "euclidean"):
        super(EuclideanCost, self).__init__(name=name)
        if isinstance(Q_diag, float):
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

    def __call__(self, 
                 state: np.ndarray,
                 control: np.ndarray,
                 observation: Dict[str, Any]) -> Union[np.ndarray, float]:
        x = extract_dims(state, self._state_dims)
        g = observation[self._goal_key]
        
        result = np.linalg.norm(x - g, axis=-1)
        if isinstance(self._Q, float):
            result = self._Q * np.einsum("...i,...j->...", result, result)
        else:
            result = np.einsum("...i,ij,...j->...", result, self._Q, result)
        if not self._squared:
            result = np.sqrt(result)
        
        return result


class ControlCost(AbstractNumPyCost):
    
    def __init__(self,
                 R_diag: Union[float, Tuple[float, ...]],
                 control_dims: Optional[Union[int, Tuple[int, ...]]] = None,
                 name: str = "control"):
        super(ControlCost, self).__init__(name=name)
        if isinstance(R_diag, float):
            assert R_diag > 0., f"R_diag must be > 0 if single value, got {R_diag}"
            if control_dims is not None:
                if isinstance(control_dims, int):
                    R_diag = np.eye(control_dims) * R_diag
                else:
                    R_diag = np.eye(len(control_dims)) * R_diag
        else:
            if control_dims is not None:
                r_len = len(R_diag)
                if isinstance(control_dims, int):
                    state_dims_len = control_dims
                else:
                    state_dims_len = len(control_dims)
                assert r_len == state_dims_len, f"Length of R_diag must match state_dims \
                    final size, got {r_len} and {state_dims_len}"
            R_diag = np.diag(R_diag) 
            assert (np.diag(R_diag) > 0.).all(), f"All elements of R_diag must be > 0"
            
        self._R = R_diag
        self._control_dims = control_dims
        
    def __call__(self, 
                 state: np.ndarray,
                 control: np.ndarray,
                 observation: Dict[str, Any]) -> Union[np.ndarray, float]:
        u = extract_dims(control, self._control_dims)
        
        if isinstance(self._R, float):
            result = self._R * np.einsum("...i,...j->...", u, u)
        else:
            result = np.einsum("...i,ij,...j->...", u, self._R, u)
        
        return result
