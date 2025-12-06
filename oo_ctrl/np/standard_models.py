import numpy as np

from typing import Tuple, Optional, Dict, Any
from oo_ctrl.np.core import AbstractNumPyModel, AbstractStateTransform
from oo_ctrl.np.util import wrap_angle


class UnicycleModel(AbstractNumPyModel):
    r"""Unicycle kinematics model.

    State :math:`\mathbf{x}` includes Cartesian coordinates :math:`(x, y)`
    in metres and orientation :math:`\theta` in radians: :math:`\mathbf{x} = [x, y, \theta]^T`.

    Control :math:`\mathbf{u}=[v, w]^T` includes linear and angular velocities in m/s and rad/s.

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


class BicycleModel(AbstractNumPyModel):
    r"""Bicycle kinematics model.

    State :math:`\mathbf{x}` includes Cartesian coordinates of rear axle center :math:`(x, y)`
    in metres and orientation :math:`\theta` in radians: :math:`\mathbf{x} = [x, y, \theta]^T`.

    Control :math:`\mathbf{u}=[v, delta]^T` includes linear velocity in m/s
    and steering angle of the front wheel in radians.

    State update is calculated as:
    :math:`x_{t+1} = x_t+ v_t * cos(\theta_t) * dt`,
    :math:`y_{t+1} = y_t + v_t * sin(\theta_t) * dt`,
    :math:`\theta_{t+1} = theta_t + (v / l) * tan(delta) * dt`,
    where dt is a discrete time step in seconds, l is the wheel base size in metres.

    Args:
        dt (float): Discrete time step
        wheel_base: Wheel base size
        linear_bounds (Tuple[float, float], optional): Bounds on linear velocity. Defaults to ``(-np.inf, np.inf)``.
        angular_bounds (Tuple[float, float], optional): Bounds on angular velocity. Defaults to ``(-np.inf, np.inf)``.
        force_clip (bool): If ``True``, the control input is clipped between lb and ub on every model call. Defaults to ``True``. 
    """

    def __init__(self,
                 dt: float,
                 wheel_base: float,
                 linear_bounds: Tuple[float, float] = (-np.inf, np.inf),
                 angular_bounds: Tuple[float, float] = (-np.inf, np.inf),
                 force_clip: bool = False):
        super(BicycleModel, self).__init__(control_lb=(linear_bounds[0], angular_bounds[0]),
                                           control_ub=(linear_bounds[1], angular_bounds[1]))
        self._dt = dt
        self._wheel_base = wheel_base
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
        delta = control[..., 1]
        l = self._wheel_base

        x_new = x + v * np.cos(theta) * self._dt
        y_new = y + v * np.sin(theta) * self._dt
        theta_new = wrap_angle(theta + (v / l) * np.tan(delta) * self._dt)

        state_new = np.concatenate((
            x_new[..., np.newaxis],
            y_new[..., np.newaxis],
            theta_new[..., np.newaxis]
        ), axis=-1)

        return state_new


class RearToCenterTransform(AbstractStateTransform):
    r"""Transformation from rear axel center to vehicle's center for bicycle model.

    Forward transform is defined as:
    :math: x := x + (l / 2) * cos(theta)
    :math: y := y + (l / 2) * sin(theta)
    :math: theta := theta
    where l is the wheel base of the bicycle model, initial state (x, y, theta) is defined
    as coordinates of the rear axle center and vehicle's orientation. Transformed state includes
    coordinates of the vehicle's center and vehicle's orientation.

    Inverse transform is defined as:
    :math: x := x - (l / 2) * cos(theta)
    :math: y := y - (l / 2) * sin(theta)
    :math: theta := theta
    where l is the wheel base of the bicycle model, initial state (x, y, theta) is defined
    as coordinates of the vehicle's center and its orientation. Transformed state includes
    coordinates of the vehicle's rear axle center and its orientation.

    Args:
        wheel_base: Wheel base size
    """

    def __init__(self, wheel_base: float) -> None:
        assert wheel_base > 0., f"Wheel base must be > 0, got {wheel_base}"
        super(RearToCenterTransform, self).__init__()
        self._wheel_base = wheel_base
        self._d = wheel_base / 2.

    def forward(self,
                state: np.ndarray,
                observation: Optional[Dict[str, Any]] = None) -> np.ndarray:
        x = state[..., 0]
        y = state[..., 1]
        theta = state[..., 2]

        x = x + self._d * np.cos(theta)
        y = y + self._d * np.sin(theta)

        state = np.concatenate((
            x[..., np.newaxis],
            y[..., np.newaxis],
            theta[..., np.newaxis]
        ), axis=-1)

        return state


    def inverse(self,
                state: np.ndarray,
                observation: Optional[Dict[str, Any]] = None) -> np.ndarray:
        x = state[..., 0]
        y = state[..., 1]
        theta = state[..., 2]

        x = x - self._d * np.cos(theta)
        y = y - self._d * np.sin(theta)

        state = np.concatenate((
            x[..., np.newaxis],
            y[..., np.newaxis],
            theta[..., np.newaxis]
        ), axis=-1)

        return state