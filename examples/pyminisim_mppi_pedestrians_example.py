
# sys.path.append('..')

import random
import time
import numpy as np
import oo_ctrl as octrl
import matplotlib.pyplot as plt

from typing import Tuple, Dict, Optional
from pyminisim.core import Simulation
from pyminisim.world_map import EmptyWorld
from pyminisim.robot import UnicycleRobotModel
from pyminisim.pedestrians import (ORCAParams, 
                                   ORCAPedestriansModel,
                                   ReplayPedestriansPolicy,
                                   RandomWaypointTracker)
from pyminisim.sensors import PedestrianDetectorConfig, PedestrianDetector
from pyminisim.visual import Renderer, CircleDrawing


SEED = 42


OBSTACLES = np.array([[1.5, 0., 0.8]])


def create_sim(vis: bool,
               ped_states: Optional[Tuple[np.ndarray, np.ndarray]]) -> Tuple[Simulation, Renderer]:
    robot_model = UnicycleRobotModel(initial_pose=np.array([0., 0., 0.]),
                                     initial_control=np.array([0., np.deg2rad(0.)]))
    sensors = [PedestrianDetector(config=PedestrianDetectorConfig(fov=2*np.pi,
                                                                  max_dist=5.,
                                                                  return_type=PedestrianDetectorConfig.RETURN_ABSOLUTE))]
    if ped_states is None:
        n_pedestrians = 7
        pedestrians = ORCAPedestriansModel(dt=0.01,
                                        waypoint_tracker=RandomWaypointTracker(world_size=(10, 10)),
                                        n_pedestrians=n_pedestrians,
                                        params=ORCAParams(),
                                        max_speeds=np.random.uniform(1., 2., (n_pedestrians,)))
    else:
        pedestrians = ReplayPedestriansPolicy(poses=ped_states[0],
                                              velocities=ped_states[1])
    sim = Simulation(sim_dt=0.01,
                     # world_map=CirclesWorld(circles=OBSTACLES),
                     world_map=EmptyWorld(),
                     robot_model=robot_model,
                     pedestrians_model=pedestrians,
                     sensors=sensors,
                     rt_factor=1. if vis else None)
    if vis:
        renderer = Renderer(simulation=sim,
                            resolution=60.0,
                            screen_size=(800, 800),
                            camera="robot")
    else:
        renderer = None
    return sim, renderer


def collect_pedestrian_trajectories() -> Tuple[np.ndarray,
                                               np.ndarray,
                                               np.ndarray]:
    sim, _ = create_sim(vis=False, ped_states=None)
    max_steps = 10 * 300
    hold_time = sim.sim_dt
    all_poses = []
    all_velocities = []
    policy_steps_poses = []
    for _ in range(max_steps):
        step_poses = sim.current_state.world.pedestrians.poses
        step_poses = np.stack([step_poses[k] for k in sorted(step_poses.keys())], axis=0)
        step_vels = sim.current_state.world.pedestrians.velocities
        step_vels = np.stack([step_vels[k] for k in sorted(step_vels.keys())], axis=0)
        all_poses.append(step_poses)
        all_velocities.append(step_vels)
        if hold_time >= 0.1:
            policy_steps_poses.append(step_poses)
            hold_time = 0.
        sim.step()
        hold_time += sim.sim_dt
        
    all_poses = np.stack(all_poses, axis=0)
    all_velocities = np.stack(all_velocities, axis=0)
    policy_steps_poses = np.stack(policy_steps_poses, axis=0)
        
    return all_poses, all_velocities, policy_steps_poses


def create_controller(horizon: int) -> octrl.np.MPPI:
    return octrl.np.MPPI(
        horizon=horizon,
        n_samples=10000,
        lmbda=10.,
        model=octrl.np.UnicycleModel(dt=0.1,
                                     linear_bounds=(0., 1.5),
                                     angular_bounds=(-np.pi / 2, np.pi / 2),
                                     force_clip=True),
        biased=False,
        cost_monitor=True,
        sampler=octrl.np.GaussianActionSampler(stds=(2., 2.)),
        cost=[
            octrl.np.EuclideanRatioGoalCost(Q=12.,
                                            squared=False,
                                            state_dims=2,
                                            name="goal"),
            # octrl.np.ControlCost(R_diag=(0.5, 0.1)),
            # octrl.np.EuclideanObstaclesCost(Q=10.,
            #                                 squared=False,
            #                                 reduction=octrl.np.Reduction.SUM_INVERSE)
            octrl.np.CollisionIndicatorCost(Q=1000.,
                                            safe_distance=0.65,
                                            name="CA")
        ]
    )


def single_run():
    ped_poses, ped_vels, ped_predictions = collect_pedestrian_trajectories()
    ped_predictions = ped_predictions.transpose((1, 0, 2))
    
    sim, renderer = create_sim(vis=True,
                               ped_states=(ped_poses, ped_vels))
    renderer.initialize()

    goal = np.array([3., -2.])
    horizon = 50
    controller = create_controller(horizon=horizon)
    renderer.draw("goal", CircleDrawing(goal, 0.1, (255, 0, 0), 0))

    sim.step()  # First step can take some time due to Numba compilation

    u_pred = np.array([0., 0.])
    hold_time = sim.sim_dt
    hold_iteration = 0

    # for _ in range(100):
    while True:
        renderer.render()
        
        if np.linalg.norm(sim.current_state.world.robot.pose[:2] - goal) < 0.3:
            break

        if hold_time >= 0.1:
            if ped_predictions[:, hold_iteration:(hold_iteration + horizon + 1), :2].shape[1] < horizon + 1:
                break

            x_current = sim.current_state.world.robot.pose
            u_pred, info = controller.step(x_current,
                                           {"goal": goal,
                                            "obstacles": ped_predictions[:, hold_iteration:(hold_iteration + horizon + 1), :2]})
            hold_time = 0.
            hold_iteration += 1
            renderer.draw("ped_pred", CircleDrawing(ped_predictions[:, hold_iteration:(hold_iteration + horizon + 1), :2].reshape((-1, 2)),
                                                    0.1, (0, 255, 0), 0))
            renderer.draw("robot_traj", CircleDrawing(info["x_seq"][:, :2], 0.1, (0, 0, 255), 0))

        sim.step(u_pred)
        # print(f"RT factor: {sim.sim_dt / (finish_time - start_time)}")
        hold_time += sim.sim_dt

    # Done! Time to quit.
    renderer.close()
    
    # monitor = controller.cost_monitor
    # plt.plot(monitor.get("goal").sum(axis=1), label="goal")
    # # plt.plot(monitor.get("control").sum(axis=1), label="control")
    # plt.plot(monitor.get("CA").sum(axis=1), label="CA")
    # plt.legend()
    # plt.show()


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    for _ in range(10):
        single_run()


if __name__ == '__main__':
    main()
