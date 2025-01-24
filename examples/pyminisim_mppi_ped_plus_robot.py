
# sys.path.append('..')

import random
import time
import numpy as np
import os
import json
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
AMOUNT_EXPERIMENTS = 1000

OBSTACLES = np.array([[1.5, 0., 0.8]])

NUM_PEDESTRIANS = 1

def create_sim(vis: bool,
               ped_states: Optional[Tuple[np.ndarray, np.ndarray]]) -> Tuple[Simulation, Renderer]:
    robot_model = UnicycleRobotModel(initial_pose=np.array([0., 0., 0.]),
                                     initial_control=np.array([0., np.deg2rad(0.)]))
    sensors = [PedestrianDetector(config=PedestrianDetectorConfig(fov=2*np.pi,
                                                                  max_dist=5.,
                                                                  return_type=PedestrianDetectorConfig.RETURN_ABSOLUTE))]
    if ped_states is None:
        n_pedestrians = NUM_PEDESTRIANS
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
    max_steps = 300
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

def collect_pedestrian_trajectories_for_mppi():
    max_steps = 10 * 300
    hold_time = 0.01
    sim_dt=0.01
    all_poses = []
    all_velocities = []
    policy_steps_poses = []
 

    waypoint_tracker=RandomWaypointTracker(world_size=(10, 10))

    random_positions = waypoint_tracker.sample_independent_points(NUM_PEDESTRIANS, 0.5)
    random_orientations = np.random.uniform(-np.pi, np.pi, size=NUM_PEDESTRIANS)
    
    initial_poses_pedestrians = np.hstack([random_positions, random_orientations.reshape(-1, 1)])
    initial_velocities_pedestrians = np.zeros((NUM_PEDESTRIANS, 2))

    initial_poses_robot = np.array([0., 0., 0.])
    initial_velocities_robot = np.array([0., 0.])

    initial_state = np.vstack([initial_poses_robot[np.newaxis, ...], initial_poses_pedestrians])
    initial_state_velocities = np.vstack([initial_velocities_robot[np.newaxis, ...], initial_velocities_pedestrians])

    if waypoint_tracker.state is None:
            waypoint_tracker.resample_all({i: initial_poses_pedestrians[i] for i in range(NUM_PEDESTRIANS)})
    all_poses.append(initial_state.reshape(-1))
    all_velocities.append(initial_state_velocities.reshape(-1))

    controller = create_controller_combined(horizon=30, n_samples=3000)

    robot_actions = []
    robot_trajectory = []

    for _ in range(max_steps-1):
        current_poses = all_poses[-1]
        # current_poses = np.stack([current_poses[k] for k in sorted(current_poses.keys())], axis=0)
        current_vels = all_velocities[-1]

        goal_robot = np.array([3., -2.])
        goal_pedestrians = waypoint_tracker.state._next_waypoints
        goal_pedestrians = np.stack([goal_pedestrians[k] for k in sorted(goal_pedestrians.keys())], axis=0)
        goal_pedestrians = goal_pedestrians[:, 0,...]
        goal_state = np.vstack([goal_robot[np.newaxis, ...], goal_pedestrians])
        goal_state = goal_state.reshape(-1)

        if hold_time >= 0.1:
            # current_poses = current_poses.reshape(-1)
            u_pred, info = controller.step(current_poses, {"goal": goal_state,
                                                            "obstacles": current_poses[3:]})
            robot_actions.append(info['u_seq'][:,:2])
            robot_trajectory.append(info['x_seq'][1:,:3])
            policy_steps_poses.append(info['x_seq'][1][3:].reshape(NUM_PEDESTRIANS,3))
            all_velocities[-1] = u_pred
            hold_time = 0.
        
        new_poses = interpolation_all_states(current_poses, current_vels, sim_dt)
        all_poses.append(new_poses)
        
        all_velocities.append(all_velocities[-1])
        
        
        hold_time += sim_dt
        waypoint_tracker.update_waypoints({i: new_poses[3*(i+1):3*(i+1)+2] for i in range(NUM_PEDESTRIANS)})

    all_poses = np.stack(all_poses, axis=0)
    all_velocities = np.stack(all_velocities, axis=0)
    policy_steps_poses = np.stack(policy_steps_poses, axis=0)
    robot_actions = np.stack(robot_actions, axis=0)
    robot_trajectory = np.stack(robot_trajectory, axis=0)
    all_pedestrians_poses = all_poses[:,3:].reshape(max_steps,NUM_PEDESTRIANS,3)
    all_pedestrians_velocities = np.zeros((max_steps,NUM_PEDESTRIANS,3))

    return all_pedestrians_poses, all_pedestrians_velocities, policy_steps_poses, robot_actions, robot_trajectory

def interpolation_all_states(poses: np.ndarray, velocities: np.ndarray, sim_dt: float) -> np.ndarray:
    x_state = poses[0::3]
    y_state = poses[1::3]
    theta_state =poses[2::3]
    v_state = velocities[0::2]
    w_state = velocities[1::2]

    x_new = x_state + v_state * np.cos(theta_state) * sim_dt
    y_new = y_state + v_state * np.sin(theta_state) * sim_dt
    theta_new = theta_state + w_state * sim_dt

    new_poses = np.stack([x_new, y_new, theta_new],axis=1)
    return new_poses.reshape(-1)

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
                                            name="goal",
                                            ),
            # octrl.np.ControlCost(R_diag=(0.5, 0.1)),
            # octrl.np.EuclideanObstaclesCost(Q=10.,
            #                                 squared=False,
            #                                 reduction=octrl.np.Reduction.SUM_INVERSE)
            octrl.np.CollisionIndicatorCost(Q=1000.,
                                            safe_distance=0.65,
                                            name="CA")
        ],
    )


def create_controller_combined(horizon: int, n_samples: int) -> octrl.np.MPPI:
    robot_stds = (2., 2.)
    pedestrian_stds = (2.,2.) * NUM_PEDESTRIANS

    robot_dims = (0, 1)  # Robot coordinates always first
    pedestrian_dims = tuple(i for ped in range(NUM_PEDESTRIANS) 
                            for i in (3 + ped * 3, 4 + ped * 3))
    state_dims = robot_dims + pedestrian_dims

    return octrl.np.MPPI(
        horizon=horizon,
        n_samples=n_samples,
        lmbda=10.,
        model=octrl.np.UnicycleModelCombined(dt=0.1,
                                     linear_bounds=(0., 1.5),
                                     angular_bounds=(-np.pi / 2, np.pi / 2),
                                     force_clip=True,
                                     n_pedestrians=NUM_PEDESTRIANS),
        biased=False,
        cost_monitor=True,
        # sampler=octrl.np.GaussianActionSampler(stds=(2., 2., 2., 2., 2.,2., 
        #                                              2., 2., 2., 2.,2., 2., 
        #                                              2., 2., 2.,2)),
        # sampler=octrl.np.GaussianActionSampler(stds=(2., 2., 2., 2., 2.,2., 
        #                                              2., 2.)),
        sampler=octrl.np.GaussianActionSampler(stds=robot_stds + pedestrian_stds),
        # u_prev=np.zeros((horizon,2 + 2*NUM_PEDESTRIANS)),
        cost=[
            octrl.np.EuclideanRatioGoalCombinedCost(Q=12.,
                                            squared=False,
                                            # TODO: create function to get state dims depending on n_pedestrians
                                            # state_dims=(0,1,3,4,6,7,9,10,12,13,15,16,18,19,21,22),
                                            # state_dims=(0,1,3,4,6,7,9,10),
                                            state_dims=state_dims,
                                            name="goal",
                                            mode="mean"),
            # octrl.np.ControlCost(R_diag=(0.5, 0.1)),
            # octrl.np.EuclideanObstaclesCost(Q=10.,
            #                                 squared=False,
            #                                 reduction=octrl.np.Reduction.SUM_INVERSE)
            # octrl.np.CollisionIndicatorCombinedCost(Q=1000.,
            #                                 safe_distance=0.65,
            #                                 name="CA",
            #                                 state_dims=24)
        ]
    )


def save_experiment(pedestrian_poses: np.ndarray, 
                    robot_action: np.ndarray, 
                    robot_goal:np.ndarray,
                    robot_poses, 
                    directory: str = "./dataset"):
    if not os.path.exists(directory):
        os.mkdir(directory)
    files = os.listdir(directory)

    if not files:
        print("No files")
        next_scenario = 1
    else:
        next_scenario = max([int(file.split('_')[1].split('.')[0]) for file in files]) + 1

    new_scene = {
            "scene_id": next_scenario, 
            "pedestrians": [
                {"pose": pose.tolist()}
                for pose in pedestrian_poses
            ],
            "robot": {
                "goal": robot_goal.tolist(),
                "action": robot_action,
                "pose": robot_poses,
            }
        }
    json.dump(new_scene, open(f"{directory}/scene_{next_scenario}.json", "w"),indent=4)

def single_run():
    actions = []
    robot_poses = []
    #ped_poses, ped_vels, ped_predictions = collect_pedestrian_trajectories()
    ped_poses, ped_vels, ped_predictions, robot_actions, robot_trajectory = collect_pedestrian_trajectories_for_mppi()
    ped_predictions = ped_predictions.transpose((1, 0, 2))
    
    sim, renderer = create_sim(vis=True,
                               ped_states=(ped_poses, ped_vels))
    renderer.initialize()

    goal = np.array([3., -2.])
    horizon = 50
    # controller = create_controller(horizon=horizon)
    renderer.draw("goal", CircleDrawing(goal, 0.1, (255, 0, 0), 0))

    sim.step()  # First step can take some time due to Numba compilation

    u_pred = np.array([0., 0.])
    hold_time = sim.sim_dt
    hold_iteration = 0

    # for _ in range(100):
    while True:
        renderer.render()
        
        if np.linalg.norm(sim.current_state.world.robot.pose[:2] - goal) < 0.3:
            # save_experiment(
            #     pedestrian_poses=ped_predictions,
            #     robot_action=actions,
            #     robot_goal=goal,
            #     robot_poses=robot_poses
            #     )
            break

        if hold_time >= 0.1:
            if ped_predictions[:, hold_iteration:(hold_iteration + horizon + 1), :2].shape[1] < horizon + 1:
                break

            x_current = sim.current_state.world.robot.pose
            # u_pred_test, info = controller.step(x_current,
            #                                {"goal": goal,
            #                                 "obstacles": ped_predictions[:, hold_iteration:(hold_iteration + horizon + 1), :2]})
            # actions.append(info['u_seq'].tolist())
            # robot_poses.append(x_current.tolist())
            u_pred = robot_actions[hold_iteration][0]
            hold_time = 0.
            hold_iteration += 1
            renderer.draw("ped_pred", CircleDrawing(ped_predictions[:, hold_iteration:(hold_iteration + horizon + 1), :2].reshape((-1, 2)),
                                                    0.1, (0, 255, 0), 0))
            renderer.draw("robot_traj", CircleDrawing(robot_trajectory[hold_iteration][:,:2], 0.1, (0, 0, 255), 0))

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
    for _ in range(AMOUNT_EXPERIMENTS):
        single_run()


if __name__ == '__main__':
    main()
