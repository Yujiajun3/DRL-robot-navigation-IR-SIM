import math
from pathlib import Path

import numpy as np
import torch
import secrets
from openai import OpenAI
from torch.utils.tensorboard import SummaryWriter
import os


class LLM(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        lr=1e-4,
        save_every=0,
        load_model=False,
        save_directory=Path("robot_nav/models/TD3/checkpoint"),
        model_name="TD3",
        load_directory=Path("robot_nav/models/TD3/checkpoint"),
    ):
        # Initialize the Actor network

        self.action_dim = action_dim
        self.max_action = max_action
        self.state_dim = state_dim
        self.writer = SummaryWriter()
        self.iter_count = 0
        if load_model:
            self.load(filename=model_name, directory=load_directory)
        self.save_every = save_every
        self.model_name = model_name
        self.save_directory = save_directory

        api_key = os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.model = "o1-mini"
        self.chat_history = []

    def get_action(self, obs, add_noise):
        if add_noise:
            return (
                self.act(obs) + np.random.normal(0, 0.2, size=self.action_dim)
            ).clip(-self.max_action, self.max_action)
        else:
            return self.act(obs)

    def act(self, state):
        # Function to get the action from the actor
        laser = state[: self.state_dim - 5]
        laser = [l.item() for l in laser]
        distance = state[-5]
        cos = state[-4]
        sin = state[-3]
        lin_vel = state[-2]
        ang_vel = state[-1]

        theta = math.atan2(sin, cos)
        assistant = (
            f"You are a circular differential drive robot with 0.2 meter radius. There is an installed 2d lidar "
            f"in 180 degree fov with {self.state_dim - 5} values measuring distance to the nearest obstacle. "
            f"Your linear velocity is limited between 0 and 0.5 m/s. "
            f"Your angular velocity is limited between -1 and 1 r/s. "
            f"Each step is executed for 0.3 seconds. "
            f"Your task is to arrive at a given goal."
            f"User will provide description of the state that the robot is in."
            f"You must return the linear and angular velocity for the robot to take."
            f"Try to limit jerkiness of the motion."
            f"Limit your answer to just the two scalar values of linear and angular velocities separated by a comma. "
        )
        user = (
            f"Lidar returned values are as follows {laser}. Distance to the goal is {distance} meters."
            f"The heading difference to the goal is {theta} radians."
            f"The last action was {lin_vel} m/s linear velocity and {ang_vel} r/s angular velocity."
            f"Give the linear and angular velocities to take."
        )
        messages = (
            [{"role": "assistant", "content": assistant}]
            + self.chat_history
            + [{"role": "user", "content": user}]
        )

        resp = self.get_response(messages=messages)
        response = resp.choices[0].message.content
        self.chat_history.append({"role": "user", "content": user})
        self.chat_history.append({"role": "assistant", "content": response})
        if len(self.chat_history) > 2:
            self.chat_history.pop(0)
            self.chat_history.pop(0)
        action = response.split(",")
        return (float(action[0]) * 4) - 1, float(action[1])

    def get_response(self, messages):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            # temperature=0.7,  # Adjusts creativity
            # max_completion_tokens=150,  # Limits response length
        )
        return response

    # training cycle
    def train(
        self,
        replay_buffer,
        iterations,
        batch_size,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
    ):
        pass

    def save(self, filename, directory):
        pass

    def load(self, filename, directory):
        pass

    def prepare_state(self, latest_scan, distance, cos, sin, collision, goal, action):
        # update the returned data from ROS into a form used for learning in the current model
        latest_scan = np.array(latest_scan)

        inf_mask = np.isinf(latest_scan)
        latest_scan[inf_mask] = 7.0

        max_bins = self.state_dim - 5
        bin_size = int(np.ceil(len(latest_scan) / max_bins))

        # Initialize the list to store the minimum values of each bin
        min_values = []

        # Loop through the data and create bins
        for i in range(0, len(latest_scan), bin_size):
            # Get the current bin
            bin = latest_scan[i : i + min(bin_size, len(latest_scan) - i)]
            # Find the minimum value in the current bin and append it to the min_values list
            min_values.append(min(bin) / 7)

        # Normalize to [0, 1] range
        distance /= 10
        lin_vel = action[0] * 2
        ang_vel = (action[1] + 1) / 2
        state = min_values + [distance, cos, sin] + [lin_vel, ang_vel]

        assert len(state) == self.state_dim
        terminal = 1 if collision or goal else 0

        return state, terminal
