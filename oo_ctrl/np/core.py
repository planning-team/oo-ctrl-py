import numpy as np

from abc import ABC, abstractmethod
from typing import Tuple, Union, Any, Dict, Optional


class AbstractNumPyModel(ABC):
    r"""Base class for the MPC transition (dynamics) models.
    
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
    def __call__(self,
                 state: np.ndarray,
                 control: np.ndarray) -> np.ndarray:
        """Calculates the next state of the system.

        Args:
            state (np.ndarray): Current state. Has shape (n_samples, state_dim) or (state_dim,).
            control (np.ndarray): Control input. Has shape (n_samples, control_dim) or (control_dim,).
        """
        pass
    
    def clip(self, control: np.ndarray) -> np.ndarray:
        return np.clip(control,
                       self._control_lb,
                       self._control_ub)


class AbstractNumPyCost(ABC):
    r"""Base class for the MPC cost functions definitions.

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
    def __call__(self,
                 state: np.ndarray,
                 control: np.ndarray,
                 observation: Dict[str, Any]) -> np.ndarray:
        """Calculates the cost function value.

        Args:
            state (np.ndarray): State of the system. Has shape (n_samples, horizon, state_dim).
            control (np.ndarray): Input of the system. Has shape (n_samples, horizon, control_dim).
            observation (Dict[str, Any]): Auxiliary observation dictionary.
            
        Returns:
            np.ndarray: Cost function values of shape (n_samples, horizon).
        """
        pass


class AbstractActionSampler(ABC):
    """Base class for the random sampler.

    This abstract class defines the interface for sampling
    random control perturbations used in MPPI optimization.
    """

    @abstractmethod
    def __call__(self, 
                 n_samples: int,
                 horizon: int,
                 observation: Optional[Dict[str, Any]]) -> np.ndarray:
        """Samples actions from some distribution.

        Args:
            n_samples (int): Number of samples (samples dimension)
            horizon (int): Length of the horizon
            observation (Optional[Dict[str, Any]]): Auxiliary observation dictionary

        Returns:
            np.ndarray: Resulting samples of size (n_samples, horizon, control_dim)
        """
        pass


class AbstractNumPyMPC(ABC):
    """Base class for implementing Model-Predictive Control (MPC) controllers.
    
    The rest of documentation is specific for each implementation.
    """
    
    @abstractmethod
    def step(current_state: np.ndarray,
             observation: Optional[Dict[str, Any]] = None,
             *args, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        pass


class AbstractStateTransform(ABC):
    """Base class for the state transformation.

    This abstract class defines the interface for the state
    transformations. They may be useful when, for example, dynamics state space
    is not convenient for cost calculation. Example for such case is bicycle model,
    where it is convenient to apply dynamics in rear axle center state space,
    but goal-reaching and collision cost functions often operate with the vehicle's center.
    """

    @abstractmethod
    def forward(self,
                state: np.ndarray,
                observation: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Forward transformation from the dynamic model's state space to
        some target state space.

        Args:
            state: A set of states to transform of shape (..., source_state_dim)
            observation (Optional[Dict[str, Any]]): Auxiliary observation dictionary

        Returns:
            np.ndarray: Transformed states of shape (..., target_state_dim)
        """
        pass

    @abstractmethod
    def inverse(self,
                state: np.ndarray,
                observation: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Inverse transformation from the target state space to the initial
        dynamic's state space

        Args:
            state: A set of states to transform of shape (..., target_state_dim)
            observation (Optional[Dict[str, Any]]): Auxiliary observation dictionary

        Returns:
            np.ndarray: Transformed states of shape (..., source_state_dim)
        """
        pass


class AbstractPresampler(ABC):
    """Base class for the pre-samping.

    This abstract class defines the interface for the pre-samplers.
    Their goal is to prepare initializing trajectories candidates
    for controllers.
    """

    @abstractmethod
    def sample(self,
               state: np.ndarray,
               observation: Optional[Dict[str, Any]] = None):
        """Method for sampling initializing trajectories candidates.

        Args:
            state (np.ndarray): Current state of the system, shape: (state_dim,)
            observation (Optional[Dict[str, Any]]): Auxiliary observation dictionary

        Returns:
            np.ndarray: Control trajectories, shape: (n_pre_samples, horizon, control_dim)
        """
        pass
