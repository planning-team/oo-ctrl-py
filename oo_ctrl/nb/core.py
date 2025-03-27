import numpy as np

from abc import ABC, abstractmethod
from typing import Tuple, Callable


class AbstractNumbaModel(ABC):
    r"""Base class for the MPC transition (dynamics) models.
    This class works like a factory, returning Numba-compliled funcion
    via "make" method.
    
    Model is a function :math:`f(x, u): x_{t+1} := f(x_t, u_t)`, where
    :math:`x` is the state, :math:`u` is the control, :math:`t` is the time step.
    
    Model also has information about the lower and upper bounds (lb and ub) on controls.
    
    Args:
        control_lb (Tuple[float, ...]): Lower bounds on control elements
        control_ub (Tuple[float, ...]): Upper bounds on control elements
    """

    def __init__(self,
                control_lb: Tuple[float, ...],
                control_ub: Tuple[float, ...]):
        len_lb = len(control_lb)
        len_ub = len(control_ub)
        assert len(control_lb) == len(control_ub), f"Lengths of lb and ub must match, got {len_lb} and {len_ub}"
        self._control_lb = np.array(control_lb)
        self._control_ub = np.array(control_ub)
    
    @property
    def control_lb(self) -> np.ndarray:
        return self._control_lb.copy()
    
    @property
    def control_ub(self) -> np.ndarray:
        return self._control_ub.copy()
    
    @abstractmethod
    def make(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """Creates a Numba-compliled function that calculates the next state of the system.

        Args of the Numba-compliled function:
            state (np.ndarray): Current state. Has shape (n_samples, state_dim).
            control (np.ndarray): Control input. Has shape (n_samples, control_dim).


        Returns:
            Callable[[np.ndarray, np.ndarray], np.ndarray]: Numba-compliled function that calculates the next state of the system.
        """
        pass


class AbstractNumbaCost(ABC):
    r"""Base class for the MPC cost functions definitions.
    This class works like a factory, returning Numba-compliled funcion
    via "make" method.

    Cost function is a function C(X, U), where
    X is the MPC state, U is the control, and output of C is a scalar value. Goal of the MPC controller is to
    minimize this cost over the horizon.
    
    Args:
        name (str): Name of the cost function
    """
    
    def __init__(self,
                 name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def make(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """Creates a Numba-compliled function that calculates the cost function value.

        Args of the Numba-compliled function:
            state (np.ndarray): State of the system. Has shape (n_samples, horizon, state_dim).
            control (np.ndarray): Input of the system. Has shape (n_samples, horizon, control_dim).
            observation (Dict[str, Any]): Auxiliary observation dictionary.

        Returns:
            Callable[[np.ndarray, np.ndarray], np.ndarray]: Numba-compliled function that calculates the cost function value.
        """
        pass
