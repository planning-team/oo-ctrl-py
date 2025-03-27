import numpy as np

from typing import Tuple, Callable
from numba import njit, prange
from oo_ctrl.nb.core import AbstractNumbaModel
from oo_ctrl.nb.util import wrap_angle


class UnicycleModel(AbstractNumbaModel):
    r"""Unicycle kinematics model factory.

    State :math:`\mathbf{x}` includes Cartesian coordinates :math:`(x, y)`
    in metres and orientation :math:`\theta` in radians: :math:`\mathbf{x} = [x, y, \theta]^T`.

    State :math:`\mathbf{u}=[v, w]^T` includes linear and angular velocities in m/s and rad/s.

    State update is calculated as:
    :math:`x_{t+1} = x_t+ v_t * cos(\theta_t) * dt`,
    :math:`y_{t+1} = y_t + v_t * sin(\theta_t) * dt`,
    :math:`\theta_{t+1} = theta_t + w_t * dt`,
    where dt is a discrete time step in seconds.

    Args:
        dt (float): Discrete time step
        linear_bounds (Tuple[float, float], optional): Bounds on linear velocity. Defaults to ``(-np.inf, np.inf)``.
        angular_bounds (Tuple[float, float], optional): Bounds on angular velocity. Defaults to ``(-np.inf, np.inf)``.
        force_clip (bool): If ``True``, the control input is clipped between lb and ub on every model call. Defaults to ``True``. 
    """

    def __init__(self,
                 dt: float,
                 linear_bounds: Tuple[float, float] = (-np.inf, np.inf),
                 angular_bounds: Tuple[float, float] = (-np.inf, np.inf),
                 force_clip: bool = False):
        super(UnicycleModel, self).__init__(control_lb=(linear_bounds[0], angular_bounds[0]),
                                            control_ub=(linear_bounds[1], angular_bounds[1]))
        self._dt = dt
        self._force_clip = force_clip

    def make(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        control_lb = self.control_lb
        control_ub = self.control_ub
        dt = self._dt
        force_clip = self._force_clip

        @njit(parallel=True)
        def _fn(state: np.ndarray, control: np.ndarray) -> np.ndarray:
            n_samples = state.shape[0]
            result = np.zeros_like(state)
            for i in prange(n_samples):
                current_state = state[i]
                current_control = control[i]

                if force_clip:
                    current_control = np.clip(current_control, control_lb, control_ub)

                result[i, 0] = current_state[0] + current_control[0] * np.cos(current_state[2]) * dt
                result[i, 1] = current_state[1] + current_control[0] * np.sin(current_state[2]) * dt
                result[i, 2] = wrap_angle(current_state[2] + current_control[1] * dt)

            return result
        
        return _fn
