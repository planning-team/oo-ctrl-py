import numpy as np
import numba as nb

from oo_ctrl.nb.standard_costs import EuclideanGoalCost
from naive_functions import random_state_goal, euclidean_cost_naive


def test_euclidean_cost_scalar():
    Q_diag=np.random.uniform(10., 100.)
    
    states, goal = random_state_goal()
    Q = np.eye(states.shape[-1]) * Q_diag
    
    cost_fn = EuclideanGoalCost(Q_diag=np.diag(Q),
                        squared=False).make()
    
    value_naive = euclidean_cost_naive(states=states,
                                        goal=goal,
                                        Q=Q,
                                        squared=False)
    observation = nb.typed.Dict()
    observation["goal"] = goal
    value = cost_fn(states, None, observation)
    
    assert np.allclose(value, value_naive)
    
    
def test_euclidean_cost_scalar_squared():
    Q_diag=np.random.uniform(10., 100.)
    
    states, goal = random_state_goal()
    Q = np.eye(states.shape[-1]) * Q_diag
    
    cost_fn = EuclideanGoalCost(Q_diag=np.diag(Q),
                        squared=True).make()
    
    value_naive = euclidean_cost_naive(states=states,
                                        goal=goal,
                                        Q=Q,
                                        squared=True)
    observation = nb.typed.Dict()
    observation["goal"] = goal
    value = cost_fn(states, None, observation)
    
    assert np.allclose(value, value_naive)


def test_euclidean_cost_diag():
    states, goal = random_state_goal()
    Q_diag=np.random.uniform(10., 100., (states.shape[-1],))
    cost_fn = EuclideanGoalCost(Q_diag=Q_diag,
                            squared=False).make()

    Q = np.diag(Q_diag)
    
    value_naive = euclidean_cost_naive(states=states,
                                        goal=goal,
                                        Q=Q,
                                        squared=False)
    observation = nb.typed.Dict()
    observation["goal"] = goal
    value = cost_fn(states, None, observation)
    
    assert np.allclose(value, value_naive)
    
    
def test_euclidean_cost_diag_squared():
    states, goal = random_state_goal()
    Q_diag=np.random.uniform(10., 100., (states.shape[-1],))
    cost_fn = EuclideanGoalCost(Q_diag=Q_diag,
                            squared=True).make()

    Q = np.diag(Q_diag)
    
    value_naive = euclidean_cost_naive(states=states,
                                        goal=goal,
                                        Q=Q,
                                        squared=True)
    observation = nb.typed.Dict()
    observation["goal"] = goal
    value = cost_fn(states, None, observation)
    
    assert np.allclose(value, value_naive)
    

def test_euclidean_cost_scalar_dims():
    Q_diag=np.random.uniform(10., 100.)
    
    states, goal = random_state_goal()
    states = states[:, :, :4]
    goal = goal[:4]
    Q = np.eye(states.shape[-1]) * Q_diag
    
    cost_fn = EuclideanGoalCost(Q_diag=np.diag(Q),
                                squared=False,
                                state_dims=4).make()

    value_naive = euclidean_cost_naive(states=states,
                                        goal=goal,
                                        Q=Q,
                                        squared=False)
    observation = nb.typed.Dict()
    observation["goal"] = goal
    value = cost_fn(states, None, observation)
    
    assert np.allclose(value, value_naive)


def test_euclidean_cost_scalar_dims_tuple():
    # TODO: Fix this test
    Q_diag=np.random.uniform(10., 100.)
    
    states, _ = random_state_goal()
    goal = np.random.uniform(10., 100., (4,))
    Q = np.eye(4) * Q_diag

    cost_fn = EuclideanGoalCost(Q_diag=np.diag(Q),
                                squared=False,
                                state_dims=(0, 1, 5, 7)).make()
    
    value_naive = euclidean_cost_naive(states=states[:, :, (0, 1, 5, 7)],
                                        goal=goal,
                                        Q=Q,
                                        squared=False)
    observation = nb.typed.Dict()
    observation["goal"] = goal
    value = cost_fn(states, None, observation)
    
    assert np.allclose(value, value_naive)
