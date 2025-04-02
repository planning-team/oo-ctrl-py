import numpy as np

from typing import Tuple
from oo_ctrl.np.util import wrap_angle


DT = 0.1


def unicycle_naive(x: np.ndarray,
                   u: np.ndarray) -> np.ndarray:
    if len(x.shape) == 1 and len(u.shape) == 1:
        single_x = True
        x = x[np.newaxis, :]
        u = u[np.newaxis, :]
    elif len(x.shape) == 1:
        single_x = False
        x = np.tile(x, (u.shape[0], 1))
    elif len(u.shape) == 1:
        single_x = False
        u = np.tile(u, (x.shape[0], 1))
    else:
        single_x = False
        assert x.shape[0] == u.shape[0]

    result = np.zeros_like(x)
    for i in range(x.shape[0]):
        result[i, 0] = x[i, 0] + u[i, 0] * np.cos(x[i, 2]) * DT
        result[i, 1] = x[i, 1] + u[i, 0] * np.sin(x[i, 2]) * DT
        result[i, 2] = wrap_angle(x[i, 2] + u[i, 1] * DT)

    if single_x:
        result = result[0]

    return result


def random_state_goal() -> Tuple[np.ndarray, np.ndarray]:
    n_samples = 3000
    horizon = 25
    dim = 10
    states = np.random.uniform(-100., 100., (n_samples, horizon, dim))
    goal = np.random.uniform(-100., 100., (dim,))
    return states, goal


def random_controls() -> np.ndarray:
    n_samples = 3000
    horizon = 25
    dim = 5
    return np.random.uniform(-100., 100., (n_samples, horizon, dim))


def random_state_obstacles(static_obstacle: bool) -> Tuple[np.ndarray, np.ndarray]:
    n_samples = 30
    horizon = 25
    n_obstacles = 10
    dim = 2
    states = np.random.uniform(-100., 100., (n_samples, horizon, dim))
    if static_obstacle:
        obstacles = np.random.uniform(-100., 100., (n_obstacles, dim))
    else:
        obstacles = np.random.uniform(-100., 100., (n_obstacles, horizon, dim))
    return states, obstacles


def euclidean_cost_naive(states: np.ndarray,
                         goal: np.ndarray,
                         Q: np.ndarray,
                         squared: bool) -> np.ndarray:
    cost_values = np.zeros(states.shape[:2])
    for i in range(states.shape[0]):
        for j in range(states.shape[1]):
            x = states[i, j]
            diff = (x - goal)
            cost_value = diff.T @ Q @ diff
            if not squared:
                cost_value = np.sqrt(cost_value)
            cost_values[i, j] = cost_value
    return cost_values


def euclidean_ratio_cost_naive(states: np.ndarray,
                         goal: np.ndarray,
                         Q: np.ndarray,
                         squared: bool) -> np.ndarray:
    cost_values = np.zeros(states.shape[:2])
    for i in range(states.shape[0]):
        dist_initial = np.linalg.norm(states[i, 0] - goal, axis=-1)
        for j in range(states.shape[1]):
            if j == 0:
                cost_value = 0.
            else:
                cost_value = np.linalg.norm(states[i, j] - goal, axis=-1) / dist_initial
                if squared:
                    cost_value = cost_value ** 2
            cost_values[i, j] = Q * cost_value
    return cost_values


def collision_indicator_cost_naive(states: np.ndarray,
                                   obstacles: np.ndarray,
                                   Q: float,
                                   safe_distance: float) -> np.ndarray:
    cost_values = np.zeros(states.shape[:2])
    for i in range(states.shape[0]):
        for j in range(states.shape[1]):
            for k in range(obstacles.shape[0]):
                x = states[i, j]
                if len(obstacles.shape) == 2:
                    diff = (x - obstacles[k])
                else:
                    diff = (x - obstacles[k, j])
                distances = np.linalg.norm(diff, axis=-1)
                collisions = (distances <= safe_distance)
                cost_values[i, j] += Q * collisions
    return cost_values 
