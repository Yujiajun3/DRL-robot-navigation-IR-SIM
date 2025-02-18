from robot_nav.models.TD3.TD3 import TD3
from robot_nav.models.SAC.SAC import SAC
from robot_nav.utils import get_buffer
from robot_nav.sim import SIM_ENV
from robot_nav.models.LLM.LLM import LLM
import pytest
import numpy as np


@pytest.mark.parametrize("model", [TD3, SAC])
def test_models(model):
    test_model = model(
        state_dim=10,
        action_dim=2,
        max_action=1,
        device="cpu",
        save_every=0,
        load_model=False,
    )  # instantiate a model

    sim = SIM_ENV

    prefilled_buffer = get_buffer(
        model=test_model,
        sim=sim,
        load_saved_buffer=True,
        pretrain=False,
        pretraining_iterations=0,
        training_iterations=0,
        batch_size=0,
        buffer_size=100,
        file_names=["test_data.yml"],
    )

    test_model.train(
        replay_buffer=prefilled_buffer,
        iterations=2,
        batch_size=8,
    )


def test_llm():
    model = LLM(
        state_dim=10,
        action_dim=2,
        max_action=1,
        device="cpu",
        save_every=0,
        load_model=False,
    )

    sim = SIM_ENV(world_file="test_world.yaml")
    latest_scan, distance, cos, sin, collision, goal, a, reward = sim.step(
        lin_velocity=0.0, ang_velocity=0.0
    )
    for i in range(100):
        state, terminal = model.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, a
        )  # get state a state representation from returned data from the environment

        action = model.get_action(np.array(state), True)  # get an action from the model
        a_in = [
            (action[0] + 1) / 4,
            action[1],
        ]  # clip linear velocity to [0, 0.5] m/s range

        latest_scan, distance, cos, sin, collision, goal, a, reward = sim.step(
            lin_velocity=a_in[0], ang_velocity=a_in[1]
        )
