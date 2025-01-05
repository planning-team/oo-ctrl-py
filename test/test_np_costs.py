import numpy as np

from typing import Tuple
from oo_ctrl.np.standard_costs import (EuclideanGoalCost,
                                       ControlCost,
                                       EuclideanObstaclesCost,
                                       Reduction)


def _random_state_goal() -> Tuple[np.ndarray, np.ndarray]:
    n_samples = 3000
    horizon = 25
    dim = 10
    states = np.random.uniform(-100., 100., (n_samples, horizon, dim))
    goal = np.random.uniform(-100., 100., (dim,))
    return states, goal


def _random_controls() -> np.ndarray:
    n_samples = 3000
    horizon = 25
    dim = 5
    return np.random.uniform(-100., 100., (n_samples, horizon, dim))


def _random_state_obstacles(static_obstacle: bool) -> Tuple[np.ndarray, np.ndarray]:
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


def _euclidean_cost_naive(states: np.ndarray,
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


def _control_cost_naive(controls: np.ndarray,
                        R: np.ndarray) -> np.ndarray:
    result = np.zeros(controls.shape[:2])
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            u = controls[i, j]
            result[i, j] = u.T @ R @ u
    return result


def _euclidean_obstacles_cost_naive(x: np.ndarray,
                                    obstacles: np.ndarray,
                                    Q: float,
                                    squared: bool,
                                    reduction: Reduction) -> np.ndarray:
    if len(obstacles.shape) == 2:
        static_obstacle = True
    else:
        static_obstacle = False
    result = np.zeros((x.shape[0], x.shape[1]))
    for i in range(x.shape[0]):
        for t in range(x.shape[1]):
            
            distances = []
            for j in range(obstacles.shape[0]):
                if static_obstacle:
                    obstacle = obstacles[j]
                else:
                    obstacle = obstacles[j, t]
                distances.append(np.linalg.norm(x[i, t] - obstacle))
            distances = np.array(distances)
            
            if reduction == Reduction.INVERSE_SUM:
                if squared:
                    distances = distances ** 2
                cost_value = 1. / np.sum(distances)
            elif reduction == Reduction.INVERSE_MEAN:
                mean = np.mean(distances)
                if squared:
                    mean = mean ** 2
                cost_value = 1. / mean
            elif reduction == Reduction.SUM_INVERSE:
                if squared:
                    distances = distances ** 2
                cost_value = np.sum(1. / distances)
            elif reduction == Reduction.MEAN_INVERSE:
                mean_inv = np.mean(1. / distances)
                if squared:
                    mean_inv = mean_inv ** 2
                cost_value = mean_inv
            elif reduction == Reduction.INVERSE_MIN:
                min_distance = np.min(distances)
                if squared:
                    min_distance = min_distance ** 2
                cost_value = 1. / min_distance
            else:
                raise ValueError(f"Unknown reduction {reduction}")
            
            result[i, t] = Q * cost_value
            
    return result


def test_euclidean_cost_scalar():
    Q_diag=np.random.uniform(10., 100.)
    cost_fn = EuclideanGoalCost(Q_diag=Q_diag,
                            squared=False)
    
    states, goal = _random_state_goal()
    Q = np.eye(states.shape[-1]) * Q_diag
    
    value_naive = _euclidean_cost_naive(states=states,
                                        goal=goal,
                                        Q=Q,
                                        squared=False)
    value = cost_fn(states, None, {"goal": goal})
    
    assert np.allclose(value, value_naive)
    
    
def test_euclidean_cost_scalar_squared():
    Q_diag=np.random.uniform(10., 100.)
    cost_fn = EuclideanGoalCost(Q_diag=Q_diag,
                            squared=True)
    
    states, goal = _random_state_goal()
    Q = np.eye(states.shape[-1]) * Q_diag
    
    value_naive = _euclidean_cost_naive(states=states,
                                        goal=goal,
                                        Q=Q,
                                        squared=True)
    value = cost_fn(states, None, {"goal": goal})
    
    assert np.allclose(value, value_naive)


def test_euclidean_cost_diag():
    states, goal = _random_state_goal()
    Q_diag=np.random.uniform(10., 100., (states.shape[-1],))
    cost_fn = EuclideanGoalCost(Q_diag=Q_diag,
                            squared=False)

    Q = np.diag(Q_diag)
    
    value_naive = _euclidean_cost_naive(states=states,
                                        goal=goal,
                                        Q=Q,
                                        squared=False)
    value = cost_fn(states, None, {"goal": goal})
    
    assert np.allclose(value, value_naive)
    
    
def test_euclidean_cost_diag_squared():
    states, goal = _random_state_goal()
    Q_diag=np.random.uniform(10., 100., (states.shape[-1],))
    cost_fn = EuclideanGoalCost(Q_diag=Q_diag,
                            squared=True)

    Q = np.diag(Q_diag)
    
    value_naive = _euclidean_cost_naive(states=states,
                                        goal=goal,
                                        Q=Q,
                                        squared=True)
    value = cost_fn(states, None, {"goal": goal})
    
    assert np.allclose(value, value_naive)
    

def test_euclidean_cost_scalar_dims():
    Q_diag=np.random.uniform(10., 100.)
    cost_fn = EuclideanGoalCost(Q_diag=Q_diag,
                            squared=False,
                            state_dims=4)
    
    states, goal = _random_state_goal()
    states = states[:, :, :4]
    goal = goal[:4]
    Q = np.eye(states.shape[-1]) * Q_diag
    
    value_naive = _euclidean_cost_naive(states=states,
                                        goal=goal,
                                        Q=Q,
                                        squared=False)
    value = cost_fn(states, None, {"goal": goal})
    
    assert np.allclose(value, value_naive)


def test_euclidean_cost_scalar_dims_tuple():
    Q_diag=np.random.uniform(10., 100.)
    cost_fn = EuclideanGoalCost(Q_diag=Q_diag,
                            squared=False,
                            state_dims=(0, 1, 5, 7))
    
    states, _ = _random_state_goal()
    goal = np.random.uniform(10., 100., (4,))
    Q = np.eye(4) * Q_diag
    
    value_naive = _euclidean_cost_naive(states=states[:, :, (0, 1, 5, 7)],
                                        goal=goal,
                                        Q=Q,
                                        squared=False)
    value = cost_fn(states, None, {"goal": goal})
    
    assert np.allclose(value, value_naive)


def test_control_cost_scalar():
    R_diag = np.random.uniform(10., 100.)
    cost_fn = ControlCost(R_diag=R_diag)
    
    control = _random_controls()
    R = np.eye(control.shape[-1]) * R_diag
    
    value_naive = _control_cost_naive(controls=control, 
                                      R=R)
    value = cost_fn(None, control, {})
    
    assert np.allclose(value, value_naive)
    
    
def test_control_cost_diag():
    control = _random_controls()
    R_diag = np.random.uniform(10., 100., (control.shape[-1]))
    cost_fn = ControlCost(R_diag=R_diag)
    
    R = np.diag(R_diag)
    
    value_naive = _control_cost_naive(controls=control, 
                                      R=R)
    value = cost_fn(None, control, {})
    
    assert np.allclose(value, value_naive)


def test_control_cost_scalar_dims():
    R_diag = np.random.uniform(10., 100.)
    cost_fn = ControlCost(R_diag=R_diag,
                          control_dims=4)
    
    control = _random_controls()
    R = np.eye(4) * R_diag
    
    value_naive = _control_cost_naive(controls=control[:, :, :4], 
                                      R=R)
    value = cost_fn(None, control, {})
    
    assert np.allclose(value, value_naive)
    
    
def test_control_cost_scalar_dims_tuple():
    R_diag = np.random.uniform(10., 100.)
    cost_fn = ControlCost(R_diag=R_diag,
                          control_dims=(0, 2, 3))
    
    control = _random_controls()
    R = np.eye(3) * R_diag
    
    value_naive = _control_cost_naive(controls=control[:, :, (0, 2, 3)], 
                                      R=R)
    value = cost_fn(None, control, {})
    
    assert np.allclose(value, value_naive)


def test_euclidean_obstacles_cost_inverse_sum():
    states, obstacles = _random_state_obstacles(static_obstacle=False)
    cost_fn = EuclideanObstaclesCost(Q=np.random.uniform(0., 1.),
                                     squared=False,
                                     reduction=Reduction.INVERSE_SUM)
    value_naive = _euclidean_obstacles_cost_naive(x=states,
                                                  obstacles=obstacles,
                                                  Q=cost_fn._Q,
                                                  squared=cost_fn._squared,
                                                  reduction=cost_fn._reduction)
    value = cost_fn(states, None, {"obstacles": obstacles})
    assert np.allclose(value, value_naive)


def test_euclidean_obstacles_cost_inverse_mean():
    states, obstacles = _random_state_obstacles(static_obstacle=False)
    cost_fn = EuclideanObstaclesCost(Q=np.random.uniform(0., 1.),
                                     squared=False,
                                     reduction=Reduction.INVERSE_MEAN)
    value_naive = _euclidean_obstacles_cost_naive(x=states,
                                                  obstacles=obstacles,
                                                  Q=cost_fn._Q,
                                                  squared=cost_fn._squared,
                                                  reduction=cost_fn._reduction)
    value = cost_fn(states, None, {"obstacles": obstacles})
    assert np.allclose(value, value_naive)
    
    
def test_euclidean_obstacles_cost_sum_inverse():
    states, obstacles = _random_state_obstacles(static_obstacle=False)
    cost_fn = EuclideanObstaclesCost(Q=np.random.uniform(0., 1.),
                                     squared=False,
                                     reduction=Reduction.SUM_INVERSE)
    value_naive = _euclidean_obstacles_cost_naive(x=states,
                                                  obstacles=obstacles,
                                                  Q=cost_fn._Q,
                                                  squared=cost_fn._squared,
                                                  reduction=cost_fn._reduction)
    value = cost_fn(states, None, {"obstacles": obstacles})
    assert np.allclose(value, value_naive)
    
    
def test_euclidean_obstacles_cost_mean_inverse():
    states, obstacles = _random_state_obstacles(static_obstacle=False)
    cost_fn = EuclideanObstaclesCost(Q=np.random.uniform(0., 1.),
                                     squared=False,
                                     reduction=Reduction.MEAN_INVERSE)
    value_naive = _euclidean_obstacles_cost_naive(x=states,
                                                  obstacles=obstacles,
                                                  Q=cost_fn._Q,
                                                  squared=cost_fn._squared,
                                                  reduction=cost_fn._reduction)
    value = cost_fn(states, None, {"obstacles": obstacles})
    assert np.allclose(value, value_naive)
    
    
def test_euclidean_obstacles_cost_inverse_min():
    states, obstacles = _random_state_obstacles(static_obstacle=False)
    cost_fn = EuclideanObstaclesCost(Q=np.random.uniform(0., 1.),
                                     squared=False,
                                     reduction=Reduction.INVERSE_MIN)
    value_naive = _euclidean_obstacles_cost_naive(x=states,
                                                  obstacles=obstacles,
                                                  Q=cost_fn._Q,
                                                  squared=cost_fn._squared,
                                                  reduction=cost_fn._reduction)
    value = cost_fn(states, None, {"obstacles": obstacles})
    assert np.allclose(value, value_naive)

def test_euclidean_obstacles_cost_inverse_sum_squared():
    states, obstacles = _random_state_obstacles(static_obstacle=False)
    cost_fn = EuclideanObstaclesCost(Q=np.random.uniform(0., 1.),
                                     squared=True,
                                     reduction=Reduction.INVERSE_SUM)
    value_naive = _euclidean_obstacles_cost_naive(x=states,
                                                  obstacles=obstacles,
                                                  Q=cost_fn._Q,
                                                  squared=cost_fn._squared,
                                                  reduction=cost_fn._reduction)
    value = cost_fn(states, None, {"obstacles": obstacles})
    assert np.allclose(value, value_naive)


def test_euclidean_obstacles_cost_inverse_mean_squared():
    states, obstacles = _random_state_obstacles(static_obstacle=False)
    cost_fn = EuclideanObstaclesCost(Q=np.random.uniform(0., 1.),
                                     squared=True,
                                     reduction=Reduction.INVERSE_MEAN)
    value_naive = _euclidean_obstacles_cost_naive(x=states,
                                                  obstacles=obstacles,
                                                  Q=cost_fn._Q,
                                                  squared=cost_fn._squared,
                                                  reduction=cost_fn._reduction)
    value = cost_fn(states, None, {"obstacles": obstacles})
    assert np.allclose(value, value_naive)
    
    
def test_euclidean_obstacles_cost_sum_inverse_squared():
    states, obstacles = _random_state_obstacles(static_obstacle=False)
    cost_fn = EuclideanObstaclesCost(Q=np.random.uniform(0., 1.),
                                     squared=True,
                                     reduction=Reduction.SUM_INVERSE)
    value_naive = _euclidean_obstacles_cost_naive(x=states,
                                                  obstacles=obstacles,
                                                  Q=cost_fn._Q,
                                                  squared=cost_fn._squared,
                                                  reduction=cost_fn._reduction)
    value = cost_fn(states, None, {"obstacles": obstacles})
    assert np.allclose(value, value_naive)
    
    
def test_euclidean_obstacles_cost_mean_inverse_squared():
    states, obstacles = _random_state_obstacles(static_obstacle=False)
    cost_fn = EuclideanObstaclesCost(Q=np.random.uniform(0., 1.),
                                     squared=True,
                                     reduction=Reduction.MEAN_INVERSE)
    value_naive = _euclidean_obstacles_cost_naive(x=states,
                                                  obstacles=obstacles,
                                                  Q=cost_fn._Q,
                                                  squared=cost_fn._squared,
                                                  reduction=cost_fn._reduction)
    value = cost_fn(states, None, {"obstacles": obstacles})
    assert np.allclose(value, value_naive)
    
    
def test_euclidean_obstacles_cost_inverse_min_squared():
    states, obstacles = _random_state_obstacles(static_obstacle=False)
    cost_fn = EuclideanObstaclesCost(Q=np.random.uniform(0., 1.),
                                     squared=True,
                                     reduction=Reduction.INVERSE_MIN)
    value_naive = _euclidean_obstacles_cost_naive(x=states,
                                                  obstacles=obstacles,
                                                  Q=cost_fn._Q,
                                                  squared=cost_fn._squared,
                                                  reduction=cost_fn._reduction)
    value = cost_fn(states, None, {"obstacles": obstacles})
    assert np.allclose(value, value_naive)

def test_euclidean_obstacles_cost_inverse_sum_static():
    states, obstacles = _random_state_obstacles(static_obstacle=True)
    cost_fn = EuclideanObstaclesCost(Q=np.random.uniform(0., 1.),
                                     squared=False,
                                     reduction=Reduction.INVERSE_SUM)
    value_naive = _euclidean_obstacles_cost_naive(x=states,
                                                  obstacles=obstacles,
                                                  Q=cost_fn._Q,
                                                  squared=cost_fn._squared,
                                                  reduction=cost_fn._reduction)
    value = cost_fn(states, None, {"obstacles": obstacles})
    assert np.allclose(value, value_naive)
