import sys
sys.path.append('..')

import time
import numpy as np
import oo_ctrl as octrl

from typing import Tuple
from pyminisim.core import Simulation
from pyminisim.world_map import EmptyWorld
from pyminisim.robot import BicycleRobotModel
from pyminisim.sensors import LidarSensor, LidarSensorConfig, SemanticDetector, SemanticDetectorConfig
from pyminisim.visual import Renderer, CircleDrawing


OBSTACLES = np.array([[1.5, 0., 0.8]])

WHEEL_BASE = 0.324


def create_sim() -> Tuple[Simulation, Renderer]:
    robot_model = BicycleRobotModel(wheel_base=WHEEL_BASE, 
                                    initial_center_pose=np.array([0., 0., 0.]),
                                    initial_control=np.array([0., np.deg2rad(0.)]))
    sensors = []
    sim = Simulation(sim_dt=0.01,
                     # world_map=CirclesWorld(circles=OBSTACLES),
                     world_map=EmptyWorld(),
                     robot_model=robot_model,
                     pedestrians_model=None,
                     sensors=sensors,
                     rt_factor=1.)
    renderer = Renderer(simulation=sim,
                        resolution=80.0,
                        screen_size=(500, 500),
                        camera="robot")
    return sim, renderer


def create_controller() -> octrl.np.MPPI:
    return octrl.np.MPPI(
        horizon=25,
        n_samples=3000,
        lmbda=0.537,
        model=octrl.np.BicycleModel(dt=0.1,
                                    wheel_base=WHEEL_BASE,
                                    linear_bounds=(0., 1.),
                                    angular_bounds=(-np.deg2rad(30.), np.deg2rad(30.))),
        biased=False,
        sampler=octrl.np.GaussianActionSampler(stds=(np.sqrt(0.05), np.sqrt(0.05))),
        cost=[
            octrl.np.SE2C2CCost(threshold_distance=0.1,
                    threshold_angle=np.deg2rad(10.),
                    weight_distance=1.5,
                    weight_angle=1.,
                    squared=False,
                    terminal_weight=20.,
                    angle_error="cos_sin")
            # octrl.np.EuclideanGoalCost(Q_diag=35.,
            #                        squared=True,
            #                        state_dims=2)
        ],
        state_transform=octrl.np.RearToCenterTransform(WHEEL_BASE)
    )


def main():
    sim, renderer = create_sim()
    renderer.initialize()

    goal = np.array([3., -2., 0.])
    controller = create_controller()
    renderer.draw("goal", CircleDrawing(goal[:2], 0.1, (255, 0, 0), 0))

    running = True
    sim.step()  # First step can take some time due to Numba compilation

    u_pred = np.array([0., 0.])
    hold_time = sim.sim_dt

    while running:
        renderer.render()

        if hold_time >= 0.1:

            x_current = sim.current_state.world.robot.pose
            u_pred, info = controller.step(x_current,
                                           {"goal": goal})
            hold_time = 0.
            renderer.draw("robot_traj", CircleDrawing(info["x_seq"][:, :2], 0.05, (252, 196, 98), 0))

        start_time = time.time()
        sim.step(u_pred)
        finish_time = time.time()
        # print(f"RT factor: {sim.sim_dt / (finish_time - start_time)}")
        hold_time += sim.sim_dt

    # Done! Time to quit.
    renderer.close()


if __name__ == '__main__':
    main()
