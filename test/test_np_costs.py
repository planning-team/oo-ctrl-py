import numpy as np

from typing import Tuple
from oo_ctrl.np.standard_costs import (EuclideanCost,
                                       ControlCost)


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


def test_euclidean_cost_scalar():
    Q_diag=np.random.uniform(10., 100.)
    cost_fn = EuclideanCost(Q_diag=Q_diag,
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
    cost_fn = EuclideanCost(Q_diag=Q_diag,
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
    cost_fn = EuclideanCost(Q_diag=Q_diag,
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
    cost_fn = EuclideanCost(Q_diag=Q_diag,
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
    cost_fn = EuclideanCost(Q_diag=Q_diag,
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
    cost_fn = EuclideanCost(Q_diag=Q_diag,
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
