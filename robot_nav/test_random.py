from robot_nav.models.TD3.TD3 import TD3
from robot_nav.models.DDPG.DDPG import DDPG
from robot_nav.models.SAC.SAC import SAC
from robot_nav.models.HCM.hardcoded_model import HCM
from robot_nav.models.PPO.PPO import PPO
from robot_nav.models.CNNTD3.CNNTD3 import CNNTD3
import statistics
import numpy as np
import tqdm
import matplotlib.pyplot as plt

import torch
from sim import SIM_ENV


def main(args=None):
    """Main testing function"""
    action_dim = 2  # number of actions produced by the model
    max_action = 1  # maximum absolute value of output actions
    state_dim = 185  # number of input values in the neural network (vector length of state input)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # using cuda if it is available, cpu otherwise
    epoch = 0  # epoch number
    max_steps = 300  # maximum number of steps in single episode
    test_scenarios = 1000

    model = CNNTD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        load_model=True,
        model_name="TD3",
    )  # instantiate a model

    sim = SIM_ENV(
        world_file="eval_world.yaml", disable_plotting=False
    )  # instantiate environment

    print("..............................................")
    print(f"Testing {test_scenarios} scenarios")
    total_reward = []
    reward_per_ep = []
    lin_actions = []
    ang_actions = []
    total_steps = 0
    col = 0
    goals = 0
    inter_rew = []
    steps_to_goal = []
    for _ in tqdm.tqdm(range(test_scenarios)):
        count = 0
        ep_reward = 0
        latest_scan, distance, cos, sin, collision, goal, a, reward = sim.reset(
            robot_state=None,
            robot_goal=None,
            random_obstacles=True,
            random_obstacle_ids=[i + 1 for i in range(6)],
        )
        done = False
        while not done and count < max_steps:
            state, terminal = model.prepare_state(
                latest_scan, distance, cos, sin, collision, goal, a
            )
            action = model.get_action(np.array(state), False)
            a_in = [(action[0] + 1) / 4, action[1]]
            lin_actions.append(a_in[0])
            ang_actions.append(a_in[1])
            latest_scan, distance, cos, sin, collision, goal, a, reward = sim.step(
                lin_velocity=a_in[0], ang_velocity=a_in[1]
            )
            ep_reward += reward
            total_reward.append(reward)
            total_steps += 1
            count += 1
            if collision:
                col += 1
            if goal:
                goals += 1
                steps_to_goal.append(count)
            done = collision or goal
            if done:
                reward_per_ep.append(ep_reward)
            if not done:
                inter_rew.append(reward)

    total_reward = np.array(total_reward)
    reward_per_ep = np.array(reward_per_ep)
    inter_rew = np.array(inter_rew)
    steps_to_goal = np.array(steps_to_goal)
    lin_actions = np.array(lin_actions)
    ang_actions = np.array(ang_actions)
    avg_step_reward = statistics.mean(total_reward)
    avg_step_reward_std = statistics.stdev(total_reward)
    avg_ep_reward = statistics.mean(reward_per_ep)
    avg_ep_reward_std = statistics.stdev(reward_per_ep)
    avg_col = col / test_scenarios
    avg_goal = goals / test_scenarios
    avg_inter_step_rew = statistics.mean(inter_rew)
    avg_inter_step_rew_std = statistics.stdev(inter_rew)
    avg_steps_to_goal = statistics.mean(steps_to_goal)
    avg_steps_to_goal_std = statistics.stdev(steps_to_goal)
    mean_lin_action = statistics.mean(lin_actions)
    lin_actions_std = statistics.stdev(lin_actions)
    mean_ang_action = statistics.mean(ang_actions)
    ang_actions_std = statistics.stdev(ang_actions)
    print(f"avg_step_reward {avg_step_reward}")
    print(f"avg_step_reward_std: {avg_step_reward_std}")
    print(f"avg_ep_reward: {avg_ep_reward}")
    print(f"avg_ep_reward_std: {avg_ep_reward_std}")
    print(f"avg_col: {avg_col}")
    print(f"avg_goal: {avg_goal}")
    print(f"avg_inter_step_rew: {avg_inter_step_rew}")
    print(f"avg_inter_step_rew_std: {avg_inter_step_rew_std}")
    print(f"avg_steps_to_goal: {avg_steps_to_goal}")
    print(f"avg_steps_to_goal_std: {avg_steps_to_goal_std}")
    print(f"mean_lin_action: {mean_lin_action}")
    print(f"lin_actions_std: {lin_actions_std}")
    print(f"mean_ang_action: {mean_ang_action}")
    print(f"ang_actions_std: {ang_actions_std}")
    print("..............................................")
    model.writer.add_scalar("test/avg_step_reward", avg_step_reward, epoch)
    model.writer.add_scalar("test/avg_step_reward_std", avg_step_reward_std, epoch)
    model.writer.add_scalar("test/avg_ep_reward", avg_ep_reward, epoch)
    model.writer.add_scalar("test/avg_ep_reward_std", avg_ep_reward_std, epoch)
    model.writer.add_scalar("test/avg_col", avg_col, epoch)
    model.writer.add_scalar("test/avg_goal", avg_goal, epoch)
    model.writer.add_scalar("test/avg_inter_step_rew", avg_inter_step_rew, epoch)
    model.writer.add_scalar(
        "test/avg_inter_step_rew_std", avg_inter_step_rew_std, epoch
    )
    model.writer.add_scalar("test/avg_steps_to_goal", avg_steps_to_goal, epoch)
    model.writer.add_scalar("test/avg_steps_to_goal_std", avg_steps_to_goal_std, epoch)
    model.writer.add_scalar("test/mean_lin_action", mean_lin_action, epoch)
    model.writer.add_scalar("test/lin_actions_std", lin_actions_std, epoch)
    model.writer.add_scalar("test/mean_ang_action", mean_ang_action, epoch)
    model.writer.add_scalar("test/ang_actions_std", ang_actions_std, epoch)
    bins = 100
    model.writer.add_histogram("test/lin_actions", lin_actions, epoch, max_bins=bins)
    model.writer.add_histogram("test/ang_actions", ang_actions, epoch, max_bins=bins)

    counts, bin_edges = np.histogram(lin_actions, bins=bins)
    fig, ax = plt.subplots()
    ax.bar(
        bin_edges[:-1], counts, width=np.diff(bin_edges), align="edge", log=True
    )  # Log scale on y-axis
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency (Log Scale)")
    ax.set_title("Histogram with Log Scale")
    model.writer.add_figure("test/lin_actions_hist", fig)

    counts, bin_edges = np.histogram(ang_actions, bins=bins)
    fig, ax = plt.subplots()
    ax.bar(
        bin_edges[:-1], counts, width=np.diff(bin_edges), align="edge", log=True
    )  # Log scale on y-axis
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency (Log Scale)")
    ax.set_title("Histogram with Log Scale")
    model.writer.add_figure("test/ang_actions_hist", fig)


if __name__ == "__main__":
    main()
