
# sys.path.append('..')

import random
import time
import numpy as np
import os
import json
import oo_ctrl as octrl
import matplotlib.pyplot as plt
from tqdm import tqdm

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
                            resolution=45.0,
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

def collect_pedestrian_trajectories_with_agents(num_pedestrians: int, 
                                                horizon: int, 
                                                n_samples: int) -> Tuple[np.ndarray,
                                                                         np.ndarray,
                                                                         np.ndarray]:
    max_steps = 10 * 300
    hold_time = 0.01
    sim_dt=0.01

    # all_poses = []
    # all_velocities = []
    # policy_steps_poses = []

    initial_poses_robot = np.array([0., 0., 0.])
    initial_velocities_robot = np.array([0., 0.])

    waypoint_tracker=RandomWaypointTracker(world_size=(5, 5), reach_distance=0.55)

    random_positions = waypoint_tracker.sample_independent_points(num_pedestrians, 0.5)
    random_orientations = np.random.uniform(-np.pi, np.pi, size=num_pedestrians)
    
    initial_poses_pedestrians = np.hstack([random_positions, random_orientations.reshape(-1, 1)])
    initial_velocities_pedestrians = np.zeros((num_pedestrians, 2))

    initial_state = np.vstack([initial_poses_robot[np.newaxis, ...], initial_poses_pedestrians])
    initial_state_velocities = np.vstack([initial_velocities_robot[np.newaxis, ...], initial_velocities_pedestrians])

    # all_current_poses = []
    # all_current_velocities = []
    policy_steps_poses = []

    # all_current_poses.append(initial_state)
    # all_current_velocities.append(initial_state_velocities)

    controllers = create_controller_with_pedestrians(horizon=horizon, n_samples=n_samples,num_pedestrians=num_pedestrians)

    all_predicted_poses = []
    all_predicted_velocities = []
    all_predicted_poses.append(np.tile(initial_state[:, np.newaxis, :], (1, horizon, 1)))
    all_predicted_velocities.append(np.tile(initial_state_velocities[:, np.newaxis, :], (1, horizon, 1)))

    all_goals = []
    robot_trajectory = []
    robot_actions = []


    if waypoint_tracker.state is None:
        waypoint_tracker.resample_all({i: initial_poses_pedestrians[i] for i in range(num_pedestrians)})

    for _ in tqdm(range(max_steps-1), desc="Simulating"):
        # step_poses = sim.current_state.world.pedestrians.poses
        # step_poses = np.stack([step_poses[k] for k in sorted(step_poses.keys())], axis=0)
        # step_vels = sim.current_state.world.pedestrians.velocities
        # step_vels = np.stack([step_vels[k] for k in sorted(step_vels.keys())], axis=0)
        # all_poses.append(step_poses)
        # all_velocities.append(step_vels)
        # if hold_time >= 0.1:
        #     policy_steps_poses.append(step_poses)
        #     hold_time = 0.
        # sim.step()
        # hold_time += sim.sim_dt

        goal_robot = np.array([3., -2.])
        goal_pedestrians = waypoint_tracker.state.current_waypoints
        goal_pedestrians = np.stack([goal_pedestrians[k] for k in sorted(goal_pedestrians.keys())], axis=0)
        goal_state = np.vstack([goal_robot[np.newaxis, ...], goal_pedestrians])
        all_goals.append(goal_state)

        current_predicted_poses = all_predicted_poses[-1].copy()
        current_predicted_velocities = all_predicted_velocities[-1].copy()

        if hold_time >= 0.1:
            # call controllers to predict new actions
            for index, controller in controllers:
                mask = np.ones((len(controllers),horizon,3), dtype=bool)
                mask[index] = False
                relative_obstacles = current_predicted_poses[mask].reshape((len(controllers)-1,horizon,3))
                u_pred, info = controller.step(current_predicted_poses[index][0], {"goal": all_goals[-1][index],
                                                                                "obstacles": relative_obstacles})
                if index == 0:
                     robot_trajectory.append(info['x_seq'][1:,:3])
                     robot_actions.append(u_pred)
                # current_predicted_poses[index] = info["x_seq"][1:,...]
                current_predicted_velocities[index] = info["u_seq"]

            # all_predicted_poses.append(current_predicted_poses)
            all_predicted_velocities.append(current_predicted_velocities)
            policy_steps_poses.append(current_predicted_poses[1:])
            hold_time = 0.

        # else:
        interpolated_poses = interpolation_all_states(current_predicted_poses, current_predicted_velocities, sim_dt)
        all_predicted_poses.append(interpolated_poses)
        all_predicted_velocities.append(current_predicted_velocities)
            
        waypoint_tracker.update_waypoints({i: all_predicted_poses[-1][i+1,0,:] for i in range(num_pedestrians)})
        hold_time += sim_dt

    all_poses = np.stack(all_predicted_poses, axis=0)
    all_velocities = np.stack(all_predicted_velocities, axis=0)
    policy_steps_poses = np.stack(policy_steps_poses, axis=0)
    robot_actions = np.stack(robot_actions, axis=0)
    robot_trajectory = np.stack(robot_trajectory, axis=0)
        
    return all_poses[:,1:,0,...], all_velocities[:,1:,0,...], policy_steps_poses[:,:,0,:], robot_trajectory, robot_actions, all_goals

def interpolation_all_states(poses: np.ndarray, velocities: np.ndarray, sim_dt: float) -> np.ndarray:
    x_state = poses[...,0]
    y_state = poses[...,1]
    theta_state =poses[...,2]
    v_state = velocities[...,0]
    w_state = velocities[...,1]

    x_new = x_state + v_state * np.cos(theta_state) * sim_dt
    y_new = y_state + v_state * np.sin(theta_state) * sim_dt
    theta_new = theta_state + w_state * sim_dt

    new_poses = np.stack([x_new, y_new, theta_new],axis=2)
    return new_poses

def create_controller(horizon: int, n_samples: int) -> octrl.np.MPPI:
    return octrl.np.MPPI(
        horizon=horizon,
        n_samples=n_samples,
        lmbda=10.,
        model=octrl.np.UnicycleModel(dt=0.1,
                                     linear_bounds=(0., 1.5),
                                     angular_bounds=(-np.pi / 2, np.pi / 2),
                                     force_clip=True),
        biased=False,
        cost_monitor=False,
        sampler=octrl.np.GaussianActionSampler(stds=(2., 2.)),
        cost=[
            octrl.np.EuclideanRatioGoalCost(Q=16.,
                                            squared=True,
                                            state_dims=2,
                                            name="goal"),
            # octrl.np.ControlCost(R_diag=(0.5, 0.1)),
            # octrl.np.EuclideanObstaclesCost(Q=10.,
            #                                 squared=False,
            #                                 reduction=octrl.np.Reduction.SUM_INVERSE)
            octrl.np.CollisionIndicatorCost(Q=1000.,
                                            safe_distance=0.85,
                                            name="CA")
        ],
    )

def create_controller_with_pedestrians(horizon: int, n_samples: int, num_pedestrians: int) -> list:
    controllers = []
    robot_controller = create_controller(horizon=horizon, n_samples=n_samples)
    controllers.append((0,robot_controller))
    for i in range(1,num_pedestrians+1):
        controllers.append((i,create_controller(horizon=horizon, n_samples=n_samples)))
    return controllers

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
    ped_poses, ped_vels, ped_predictions, robot_trajectory, robot_actions,all_goals  = collect_pedestrian_trajectories_with_agents(num_pedestrians=7,horizon=50,n_samples=10000)
    ped_predictions = ped_predictions.transpose((1, 0, 2))
    ped_vels = np.zeros_like(ped_poses)
    sim, renderer = create_sim(vis=True,
                               ped_states=(ped_poses, ped_vels))
    renderer.initialize()

    goal = np.array([3., -2.])
    horizon = 50
    n_samples = 10000   
    # controller = create_controller(horizon=horizon, n_samples=n_samples)
    renderer.draw("goal", CircleDrawing(goal, 0.1, (255, 0, 0), 0))

    sim.step()  # First step can take some time due to Numba compilation

    u_pred = np.array([0., 0.])
    hold_time = sim.sim_dt
    hold_iteration = 0
    iteration = 0
    goal_robot = np.array([3., -2.])
    # for _ in range(100):
    while True:
        renderer.render()
        # Define different colors for each pedestrian and their goals
        ped_colors = [
            (128, 128, 128),
            (255, 0, 0),   # Red
            (0, 255, 0),   # Green 
            (0, 0, 255),   # Blue
            (255, 255, 0), # Yellow
            (255, 0, 255), # Magenta
            (0, 255, 255), # Cyan
            (128, 0, 0),   # Dark red
            (0, 128, 0),   # Dark green
        ]

        # Draw goals with different colors for each pedestrian
        for ped_idx, goal in enumerate(all_goals[iteration]):
            color = ped_colors[ped_idx % len(ped_colors)]
            renderer.draw(f"goals_{ped_idx}", CircleDrawing(goal, 0.1, color, 0))

        if np.linalg.norm(sim.current_state.world.robot.pose[:2] - goal_robot,2) < 0.45:
            # save_experiment(
            #     pedestrian_poses=ped_predictions,
            #     robot_action=actions,
            #     robot_goal=goal,
            #     robot_poses=robot_poses
            #     )
            break
        else:
            print(f"Distance to goal: {np.linalg.norm(sim.current_state.world.robot.pose[:2] - goal_robot,2)}")

        if hold_time >= 0.1:
            if ped_predictions[:, hold_iteration:(hold_iteration + horizon + 1), :2].shape[1] < horizon + 1:
                break

            x_current = sim.current_state.world.robot.pose
            u_pred = robot_actions[hold_iteration]

            hold_time = 0.
            hold_iteration += 1
            
            # Draw predicted trajectories with different colors for each pedestrian
            for ped_idx in range(ped_predictions.shape[0]):
                ped_traj = ped_predictions[ped_idx, hold_iteration:(hold_iteration + horizon + 1), :2]
                color = ped_colors[ped_idx+1 % len(ped_colors)]
                renderer.draw(f"ped_pred_{ped_idx}", CircleDrawing(ped_traj, 0.1, color, 0))
                
            renderer.draw("robot_traj", CircleDrawing(robot_trajectory[hold_iteration][:,:2], 0.1, (128, 128, 128), 0))
        sim.step(u_pred)
        # print(f"RT factor: {sim.sim_dt / (finish_time - start_time)}")
        hold_time += sim.sim_dt
        iteration+=1
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
