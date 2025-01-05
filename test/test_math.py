import numpy as np

from typing import Tuple, Optional
from oo_ctrl.np.util import vec_mat_vec


def _random_vec_mat() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_samples = 3000
    horizon = 25
    dim = 10
    vec_l = np.random.uniform(-10., 10., (n_samples, horizon, dim))
    mat = np.random.uniform(-10., 10., (dim, dim))
    vec_r = np.random.uniform(-10., 10., (n_samples, horizon, dim))
    return vec_l, mat, vec_r


def _vec_mat_vec_naive(vec_l: np.ndarray,
                       mat: np.ndarray,
                       vec_r: np.ndarray) -> np.ndarray:
    result = np.zeros((vec_l.shape[0], vec_l.shape[1]))
    for i in range(vec_l.shape[0]):
        for j in range(vec_l.shape[1]):
            x = vec_l[i, j]
            u = vec_r[i, j]
            result[i, j] = x.T @ mat @ u
    return result


def _vec_vec_naive(vec_l: np.ndarray,
                       vec_r: np.ndarray) -> np.ndarray:
    result = np.zeros((vec_l.shape[0], vec_l.shape[1]))
    for i in range(vec_l.shape[0]):
        for j in range(vec_l.shape[1]):
            x = vec_l[i, j]
            u = vec_r[i, j]
            result[i, j] = x.T @ u
    return result


def _mppi_tensordot_naive(w: np.ndarray,
                          epsilon: np.ndarray) -> np.ndarray:
    # w: (n_samples,)
    # epsilon: (n_samples, horizon, dim)
    result = np.zeros(epsilon.shape[1:])
    for t in range(epsilon.shape[1]):
        for d in range(epsilon.shape[2]):
            for k in range(epsilon.shape[0]):
                result[t, d] += w[k] * epsilon[k, t, d]
    return result


def test_vec_mat_vec_symmetric():
    vec_l, mat, _ = _random_vec_mat()
    values_naive = _vec_mat_vec_naive(vec_l, mat, vec_l)
    values = vec_mat_vec(vec_l, mat, vec_l)
    assert np.allclose(values, values_naive)


def test_vec_mat_vec_asymmetric():
    vec_l, mat, vec_r = _random_vec_mat()
    values_naive = _vec_mat_vec_naive(vec_l, mat, vec_r)
    values = vec_mat_vec(vec_l, mat, vec_r)
    assert np.allclose(values, values_naive)


def test_vec_vec_symmetric():
    vec_l, _, _ = _random_vec_mat()
    values_naive = _vec_vec_naive(vec_l, vec_l)
    values = vec_mat_vec(vec_l, None, vec_l)
    assert np.allclose(values, values_naive)
    
    
def test_vec_vec_asymmetric():
    vec_l, _, vec_r = _random_vec_mat()
    values_naive = _vec_vec_naive(vec_l, vec_r)
    values = vec_mat_vec(vec_l, None, vec_r)
    assert np.allclose(values, values_naive)


def test_mppi_tensordot():
    w = np.random.uniform(0., 10., (3000,))
    epsilon = np.random.uniform(-10., 10., (3000, 25, 7))
    values_naive = _mppi_tensordot_naive(w, epsilon)
    values = np.tensordot(w, epsilon, axes=(0, 0))
    assert np.allclose(values, values_naive)
