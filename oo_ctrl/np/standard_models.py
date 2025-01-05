import numpy as np

from typing import Tuple
from oo_ctrl.np.core import AbstractNumPyModel
from oo_ctrl.np.util import wrap_angle


class UnicycleModel(AbstractNumPyModel):
    r"""Unicycle kinematics model.

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

    def __call__(self,
                 state: np.ndarray,
                 control: np.ndarray) -> np.ndarray:
        if self._force_clip:
            control = self.clip(control)

        x = state[..., 0]
        y = state[..., 1]
        theta = state[..., 2]
        v = control[..., 0]
        w = control[..., 1]

        x_new = x + v * np.cos(theta) * self._dt
        y_new = y + v * np.sin(theta) * self._dt
        theta_new = wrap_angle(theta + w * self._dt)

        state_new = np.concatenate((
            x_new[..., np.newaxis],
            y_new[..., np.newaxis],
            theta_new[..., np.newaxis]
        ), axis=-1)

        return state_new
