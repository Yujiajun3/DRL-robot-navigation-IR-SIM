import irsim
import numpy as np
import random

from robot_nav.SIM_ENV.sim_env import SIM_ENV


class SIM(SIM_ENV):
    """
    A simulation environment interface for robot navigation using IRSim.

    This class wraps around the IRSim environment and provides methods for stepping,
    resetting, and interacting with a mobile robot, including reward computation.

    Attributes:
        env (object): The simulation environment instance from IRSim.
        robot_goal (np.ndarray): The goal position of the robot.
    """

    def __init__(self, world_file="robot_world.yaml", disable_plotting=False):
        """
        Initialize the simulation environment.

        Args:
            world_file (str): Path to the world configuration YAML file.
            disable_plotting (bool): If True, disables rendering and plotting.
        """
        display = False if disable_plotting else True
        self.env = irsim.make(
            world_file, disable_all_plot=disable_plotting, display=display
        )
        robot_info = self.env.get_robot_info(0)
        self.robot_goal = robot_info.goal

        self.prev_action = [0.0, 0.0]
        self.prev_state = np.array([[0.0], [0.0], [0.0]])

    def step(self, lin_velocity=0.0, ang_velocity=0.1):
        """
        Perform one step in the simulation using the given control commands.

        Args:
            lin_velocity (float): Linear velocity to apply to the robot.
            ang_velocity (float): Angular velocity to apply to the robot.

        Returns:
            (tuple): Contains the latest LIDAR scan, distance to goal, cosine and sine of angle to goal,
                   collision flag, goal reached flag, applied action, and computed reward.
        """
        self.env.step(action_id=0, action=np.array([[lin_velocity], [ang_velocity]]))
        self.env.render()

        scan = self.env.get_lidar_scan()
        latest_scan = scan["ranges"]

        robot_state = self.env.get_robot_state()
        goal_vector = [
            self.robot_goal[0].item() - robot_state[0].item(),
            self.robot_goal[1].item() - robot_state[1].item(),
        ]
        distance = np.linalg.norm(goal_vector)
        goal = self.env.robot.arrive
        pose_vector = [np.cos(robot_state[2]).item(), np.sin(robot_state[2]).item()]
        cos, sin = self.cossin(pose_vector, goal_vector)
        collision = self.env.robot.collision
        action = [lin_velocity, ang_velocity]
        reward = self.get_reward(goal, collision, action, latest_scan, lin_velocity, pose_vector, goal_vector, distance, self.prev_action, robot_state, self.prev_state)
        self.prev_action = action
        self.prev_state = robot_state

        return latest_scan, distance, cos, sin, collision, goal, action, reward

    def reset(
        self,
        robot_state=None,
        robot_goal=None,
        random_obstacles=True,
        random_obstacle_ids=None,
    ):
        """
        Reset the simulation environment, optionally setting robot and obstacle states.

        Args:
            robot_state (list or None): Initial state of the robot as a list of [x, y, theta, speed].
            robot_goal (list or None): Goal state for the robot.
            random_obstacles (bool): Whether to randomly reposition obstacles.
            random_obstacle_ids (list or None): Specific obstacle IDs to randomize.

        Returns:
            (tuple): Initial observation after reset, including LIDAR scan, distance, cos/sin,
                   and reward-related flags and values.
        """
        if robot_state is None:
            robot_state = [[random.uniform(1, 59)], [random.uniform(1, 59)], [0]]

        self.env.robot.set_state(
            state=np.array(robot_state),
            init=True,
        )

        if random_obstacles:
            if random_obstacle_ids is None:
                random_obstacle_ids = [i + 1 for i in range(7)]
            self.env.random_obstacle_position(
                range_low=[0, 0, -3.14],
                range_high=[60, 60, 3.14],
                ids=random_obstacle_ids,
                non_overlapping=True,
            )

        if robot_goal is None:
            self.env.robot.set_random_goal(
                obstacle_list=self.env.obstacle_list,
                init=True,
                range_limits=[[1, 1, -3.141592653589793], [59, 59, 3.141592653589793]],
            )
        else:
            self.env.robot.set_goal(np.array(robot_goal), init=True)
        self.env.reset()
        self.robot_goal = self.env.robot.goal

        action = [0.0, 0.0]
        latest_scan, distance, cos, sin, _, _, action, reward = self.step(
            lin_velocity=action[0], ang_velocity=action[1]
        )
        return latest_scan, distance, cos, sin, False, False, action, reward

    @staticmethod
    def get_reward(goal, collision, action, laser_scan, lin_velocity, pose_vector, goal_vector, distance, prev_action, robot_state, prev_state):
        """
        Calculate the reward for the current step.

        Args:
            goal (bool): Whether the goal has been reached.
            collision (bool): Whether a collision occurred.
            action (list): The action taken [linear velocity, angular velocity].
            laser_scan (list): The LIDAR scan readings.

        Returns:
            (float): Computed reward for the current state.
        """
        if goal:
            return 10.0
        elif collision:
            return -2.0
        else:
            radius = 0.5
            r3 = lambda x: radius - x if x < radius else 0.0
            # return action[0] - abs(action[1]) / 2 - r3(min(laser_scan)) / 2 + 0.2 / (0.1+distance) +\
            #         0.2 * lin_velocity * (pose_vector[0] * goal_vector[0] + pose_vector[1] * goal_vector[1]) / distance

            pi=3.1415926
            smoothness = -((action[0]*pose_vector[0]-prev_action[0]*np.cos(prev_state[2]).item())**2 + \
                           (action[0]*pose_vector[1]-prev_action[0]*np.sin(prev_state[2]).item())**2)

            velocity_proximity = 0.0
            if distance > 0.3:
                velocity_proximity = 1.5 * lin_velocity * (pose_vector[0] * goal_vector[0] + pose_vector[1] * goal_vector[1]) / distance
            else:
                velocity_proximity = 0.6
            return  - abs(action[1])**3*0.1 - r3(min(laser_scan)) / 400 +  0.1 * smoothness + 0.1 *  velocity_proximity



