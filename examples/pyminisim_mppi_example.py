
# sys.path.append('..')

import time
import numpy as np
import oo_ctrl as octrl

from typing import Tuple
from pyminisim.core import Simulation
from pyminisim.world_map import EmptyWorld
from pyminisim.robot import UnicycleRobotModel
from pyminisim.sensors import LidarSensor, LidarSensorConfig, SemanticDetector, SemanticDetectorConfig
from pyminisim.visual import Renderer, CircleDrawing


OBSTACLES = np.array([[1.5, 0., 0.8]])


def create_sim() -> Tuple[Simulation, Renderer]:
    robot_model = UnicycleRobotModel(initial_pose=np.array([0., 0., 0.]),
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
        lmbda=10.,
        model=octrl.np.UnicycleModel(dt=0.1,
                                     linear_bounds=(0., 1.5),
                                     angular_bounds=(-np.pi / 4, np.pi / 4)),
        biased=False,
        sampler=octrl.np.GaussianActionSampler(stds=(0.7, np.pi / 2)),
        cost=[
            octrl.np.EuclideanCost(Q_diag=35.,
                                   squared=True,
                                   state_dims=2),
            octrl.np.ControlCost(R_diag=(5., 1.))
        ]
    )


def main():
    sim, renderer = create_sim()
    renderer.initialize()

    goal = np.array([3., -2.])
    controller = create_controller()
    renderer.draw("goal", CircleDrawing(goal, 0.1, (255, 0, 0), 0))

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

        start_time = time.time()
        sim.step(u_pred)
        finish_time = time.time()
        # print(f"RT factor: {sim.sim_dt / (finish_time - start_time)}")
        hold_time += sim.sim_dt

    # Done! Time to quit.
    renderer.close()


if __name__ == '__main__':
    main()
