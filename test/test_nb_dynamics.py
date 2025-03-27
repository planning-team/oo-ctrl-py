import numpy as np

from typing import Callable
from oo_ctrl.nb.standard_models import UnicycleModel
from naive_functions import unicycle_naive, DT


def _create_unicycle_model(force_clip: bool) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    return UnicycleModel(dt=DT,
                         linear_bounds=(-1.3, 2.5),
                         angular_bounds=(-np.pi / 4, np.pi / 1.4),
                         force_clip=force_clip).make()


def test_unicycle_samples():
    x = np.random.uniform(-10., 10., (3000, 3,))
    u = np.random.uniform(-3., 3., (3000, 2,))
    model = _create_unicycle_model(force_clip=False)

    values_naive = unicycle_naive(x, u)
    values = model(x, u)

    assert np.allclose(values, values_naive)
