import numpy as np

from typing import Union, Dict, Any, Tuple, Optional
from enum import Enum
from oo_ctrl.np.core import AbstractNumPyCost
from oo_ctrl.np.util import extract_dims, wrap_angle


class Reduction(Enum):
    INVERSE_SUM = "inverse_sum"
    INVERSE_MEAN = "inverse_mean"
    SUM_INVERSE = "sum_inverse"
    MEAN_INVERSE = "mean_inverse"
    INVERSE_MIN = "inverse_min"


class EuclideanGoalCost(AbstractNumPyCost):
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

    def __call__(self, 
                 state: np.ndarray,
                 control: np.ndarray,
                 observation: Dict[str, Any]) -> Union[np.ndarray, float]:
        x = extract_dims(state, self._state_dims)
        g = observation[self._goal_key]
        
        result = x - g
        if isinstance(self._Q, float):
            result = self._Q * np.einsum("...i,...i->...", result, result)
        else:
            result = np.einsum("...i,ii,...i->...", result, self._Q, result)
        if not self._squared:
            result = np.sqrt(result)
        
        return result


class EuclideanRatioGoalCost(AbstractNumPyCost):

    def __init__(self,
                 Q: float,
                 squared: bool,
                 state_dims: Optional[Union[int, Tuple[int, ...]]] = None,
                 goal_key: str = "goal",
                 name: str = "euclidean_ratio_goal"):
        super(EuclideanRatioGoalCost, self).__init__(name=name)
        self._Q = Q
        self._squared = squared
        self._state_dims = state_dims
        self._goal_key = goal_key

    def __call__(self, 
                 state: np.ndarray,
                 control: np.ndarray,
                 observation: Dict[str, Any]) -> Union[np.ndarray, float]:
        x = extract_dims(state, self._state_dims)  # (n_samples, H, dim)
        g = observation[self._goal_key]  # (dim,)
        
        dist_all = np.linalg.norm(x - g, axis=-1)  # (n_samples, H)
        dist_initial = dist_all[:, 0, np.newaxis] # (n_samples, 1)
        
        ratio = dist_all / dist_initial  # (n_samples, H)
        if ratio.shape[1] > 1:
            ratio[:, 0] = 0.
            
        if self._squared:
            ratio = ratio ** 2
            
        return self._Q * ratio


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
            result = self._R * np.einsum("...i,...i->...", u, u)
        else:
            result = np.einsum("...i,ii,...i->...", u, self._R, u)
        
        return result


class EuclideanObstaclesCost(AbstractNumPyCost):
    """Cost function for avoiding obstacles based on Euclidean distance.

    This class implements a cost function that penalizes being close to obstacles by computing
    Euclidean distances between the system state and obstacle positions. The cost increases as
    the system gets closer to obstacles.

    Args:
        Q (float): Weight coefficient for the cost
        squared (bool): Whether to square the distances before applying reduction
        reduction (Union[str, Reduction]): Method for reducing distances to multiple obstacles:
            - "inverse_sum": 1 / sum(distances) - sensitive to being close to any obstacle
            - "inverse_mean": 1 / mean(distances) - averages effect of all obstacles
            - "sum_inverse": sum(1 / distances) - strongly penalizes being very close to any obstacle
        state_dims (Optional[Union[int, Tuple[int, ...]]]): Which dimensions of state to use for distance calculation.
            Defaults to 2 (e.g. x,y position)
        obstacles_key (str): Key for accessing obstacles in the observation dictionary. Defaults to "obstacles"
        name (str): Name of the cost component. Defaults to "euclidean_obstacles"
    """

    def __init__(self,
                 Q: float,
                 squared: bool,
                 reduction: Union[str, Reduction],
                 state_dims: Optional[Union[int, Tuple[int, ...]]] = 2,
                 obstacles_key: str = "obstacles",
                 name: str = "euclidean_obstacles"):
        super(EuclideanObstaclesCost, self).__init__(name=name)
        self._Q = Q
        if isinstance(reduction, str):
            reduction = Reduction(reduction)
        self._reduction = reduction
        self._squared = squared
        self._state_dims = state_dims
        self._obstacles_key = obstacles_key

    def __call__(self, 
                 state: np.ndarray,
                 control: np.ndarray,
                 observation: Dict[str, Any]) -> Union[np.ndarray, float]:
        obstacles = observation[self._obstacles_key] # (n_obstacles, H, dim)
        x = extract_dims(state, self._state_dims) # (n_samples, H, dim)
        
        # If obstacles don't have horizon dimension (static obstacles),
        # just copy them along the horizon
        if len(obstacles.shape) == 2:
            obstacles = np.stack([obstacles for _ in range(x.shape[1])], axis=1)
        
        x = x[:, np.newaxis, :, :]  # (n_samples, 1, H, dim)
        obstacles = obstacles[np.newaxis, :, :, :]  # (1, n_obstacles, H, dim)
        diff = x - obstacles
        distances = np.linalg.norm(diff, axis=-1)  # (n_samples, n_obstacles, H)
        
        if self._reduction == Reduction.INVERSE_SUM:
            if self._squared:
                distances = distances ** 2
            return self._Q * (1. / np.sum(distances, axis=1))
        
        if self._reduction == Reduction.INVERSE_MEAN:
            mean_distances = np.mean(distances, axis=1)
            if self._squared:
                mean_distances = mean_distances ** 2
            return self._Q * (1. / mean_distances)
        
        if self._reduction == Reduction.SUM_INVERSE:
            if self._squared:
                distances = distances ** 2
            inv_distances = 1. / distances
            return self._Q * np.sum(inv_distances, axis=1)

        if self._reduction == Reduction.MEAN_INVERSE:
            mean_inv_distances = np.mean(1. / distances, axis=1)
            if self._squared:
                mean_inv_distances = mean_inv_distances ** 2
            return self._Q * mean_inv_distances
        
        if self._reduction == Reduction.INVERSE_MIN:
            min_distances = np.min(distances, axis=1)
            if self._squared:
                min_distances = min_distances ** 2
            return self._Q * (1. / min_distances)

        raise ValueError(f"Unknown reduction {self._reduction}")


class CollisionIndicatorCost(AbstractNumPyCost):
    
    def __init__(self,
                 Q: float,
                 safe_distance: float,
                 state_dims: Optional[Union[int, Tuple[int, ...]]] = 2,
                 obstacles_key: str = "obstacles",
                 name: str = "collision_indicator"):
        super(CollisionIndicatorCost, self).__init__(name=name)
        self._Q = Q
        self._safe_distance = safe_distance
        self._state_dims = state_dims
        self._obstacles_key = obstacles_key
    
    def __call__(self, 
                 state: np.ndarray,
                 control: np.ndarray,
                 observation: Dict[str, Any]) -> Union[np.ndarray, float]:
        obstacles = observation[self._obstacles_key] # (n_obstacles, H, dim)
        x = extract_dims(state, self._state_dims) # (n_samples, H, dim)
        
        # If obstacles don't have horizon dimension (static obstacles),
        # just copy them along the horizon
        if len(obstacles.shape) == 2:
            obstacles = np.stack([obstacles for _ in range(x.shape[1])], axis=1)
        
        x = x[:, np.newaxis, :, :]  # (n_samples, 1, H, dim)
        obstacles = obstacles[np.newaxis, :, :, :]  # (1, n_obstacles, H, dim)
        diff = x - obstacles
        distances = np.linalg.norm(diff, axis=-1)  # (n_samples, n_obstacles, H)

        collisions = (distances <= self._safe_distance) # (n_samples, n_obstacles, H)
        
        reduced_cost = np.sum(collisions, axis=1)
        return self._Q * reduced_cost


class SE2C2CCost(AbstractNumPyCost):

    ANGLE_ERROR_DIFFERENCE = "difference"
    ANGLE_ERROR_COS_SIN = "cos_sin"

    def __init__(self, 
                 threshold_distance: float,
                 threshold_angle: float,
                 weight_distance: float,
                 weight_angle: float,
                 squared: bool,
                 terminal_weight: float,
                 angle_error: str = "difference",
                 goal_key: str = "goal",
                 name: str = "c2c_goal"):
        assert angle_error in (SE2C2CCost.ANGLE_ERROR_DIFFERENCE,
                               SE2C2CCost.ANGLE_ERROR_COS_SIN)
        super(SE2C2CCost, self).__init__(name=name)
        self._threshold_distance = threshold_distance
        self._threshold_angle = threshold_angle
        self._weight_distance = weight_distance
        self._weight_angle = weight_angle
        self._angle_error = angle_error
        self._squared = squared
        self._terminal_weight = terminal_weight
        self._goal_key = goal_key

    def __call__(self, 
                 state: np.ndarray,
                 control: np.ndarray,
                 observation: Dict[str, Any]) -> Union[np.ndarray, float]:
        g = observation[self._goal_key] # (state_dim,)

        metric_dists = np.linalg.norm(state[..., :2] - g[:2], axis=-1)
        angle_dists = wrap_angle(state[..., 2] - g[2])
        # Per-state flags
        reach_mask = np.logical_and(
            metric_dists <= self._threshold_distance,
            angle_dists <= self._threshold_angle
        ) # (n_samples, horizon)
        # Make all states after goal reach as goal reached
        reach_mask = np.cumsum(reach_mask, axis=1) > 1
        # Invert mask to compute costs before goal reach
        nonreach_mask = np.logical_not(reach_mask).astype(state.dtype)
        reach_mask = reach_mask.astype(state.dtype)

        if self._angle_error == SE2C2CCost.ANGLE_ERROR_DIFFERENCE:
            angle_error_sqr = angle_dists ** 2
        elif self._angle_error == SE2C2CCost.ANGLE_ERROR_COS_SIN:
            angle_error_sqr = 2. * (1. - np.cos(state[..., 2] - g[2]))
        dist_error_sqr = metric_dists ** 2

        stage_costs = self._weight_distance * dist_error_sqr + \
            self._weight_angle * angle_error_sqr
        if not self._squared:
            stage_costs = np.sqrt(stage_costs)
        stage_costs = stage_costs * nonreach_mask

        terminal_cost = self._terminal_weight * np.min(stage_costs, axis=1)
        stage_costs[:, -1] = terminal_cost

        return stage_costs
