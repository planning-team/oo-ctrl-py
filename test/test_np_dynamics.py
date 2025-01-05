import numpy as np

from oo_ctrl.np.standard_models import UnicycleModel
from oo_ctrl.np.util import wrap_angle


_DT = 0.1


def _unicycle_naive(x: np.ndarray,
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
        result[i, 0] = x[i, 0] + u[i, 0] * np.cos(x[i, 2]) * _DT
        result[i, 1] = x[i, 1] + u[i, 0] * np.sin(x[i, 2]) * _DT
        result[i, 2] = wrap_angle(x[i, 2] + u[i, 1] * _DT)
    
    if single_x:
        result = result[0]
        
    return result


def _create_unicycle_model(force_clip: bool) -> UnicycleModel:
    return UnicycleModel(dt=_DT,
                         linear_bounds=(-1.3, 2.5),
                         angular_bounds=(-np.pi / 4, np.pi / 1.4),
                         force_clip=force_clip)


def test_unicycle_single():
    x = np.random.uniform(-10., 10., (3,))
    u = np.random.uniform(-3., 3., (2,))
    model = _create_unicycle_model(force_clip=False)
    
    values_naive = _unicycle_naive(x, u)
    values = model(x, u)
    
    assert np.allclose(values, values_naive)


def test_unicycle_samples():
    x = np.random.uniform(-10., 10., (3000, 3,))
    u = np.random.uniform(-3., 3., (3000, 2,))
    model = _create_unicycle_model(force_clip=False)
    
    values_naive = _unicycle_naive(x, u)
    values = model(x, u)
    
    assert np.allclose(values, values_naive)
    
    
def test_unicycle_many_states():
    x = np.random.uniform(-10., 10., (3000, 3,))
    u = np.random.uniform(-3., 3., (2,))
    model = _create_unicycle_model(force_clip=False)
    
    values_naive = _unicycle_naive(x, u)
    values = model(x, u)
    
    assert np.allclose(values, values_naive)


def test_unicycle_many_controls():
    x = np.random.uniform(-10., 10., (3,))
    u = np.random.uniform(-3., 3., (3000, 2))
    model = _create_unicycle_model(force_clip=False)
    
    values_naive = _unicycle_naive(x, u)
    values = model(x, u)
    
    assert np.allclose(values, values_naive)
