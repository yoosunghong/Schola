# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

from typing import Literal
import pytest
import gymnasium as gym
import numpy as np
from schola.sb3.utils import VecMergeDictActionWrapper, save_model_as_onnx
import stable_baselines3 as sb3
from schola.core.spaces import DiscreteSpace, BoxSpace, DictSpace
from schola.sb3.action_space_patch import PatchedPPO

ActionSpaceType = Literal["discrete", "continuous", "both"]


@pytest.fixture
def env_class(request):
    action_space_type: ActionSpaceType = request.param

    # test env is a gym env with a dictionary observation space and dictionary action space
    class TestEnv(gym.Env):
        def __init__(self):
            self.observation_space = DictSpace(
                {
                    "image": BoxSpace(low=0, high=1, shape=(84, 84, 3)),
                    "vector": BoxSpace(low=-1, high=1, shape=(3,)),
                }
            )
            if action_space_type == "discrete":
                self.action_space = DictSpace(
                    {
                        "action": DiscreteSpace(2),
                    }
                )
            elif action_space_type == "continuous":
                self.action_space = DictSpace(
                    {"action": BoxSpace(low=-1, high=1, shape=(3,))}
                )
            else:
                self.action_space = DictSpace(
                    {
                        "action1": DiscreteSpace(2),
                        "action2": BoxSpace(low=-1, high=1, shape=(3,)),
                    }
                )
            super().__init__()

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            observation = self.observation_space.sample()
            info = {}
            return observation, info

        def step(self, action):
            observation = self.observation_space.sample()
            reward = 0
            terminated = False
            truncated = False
            info = {}
            return observation, reward, terminated, truncated, info

    return TestEnv


from stable_baselines3 import PPO, SAC, TD3, DDPG, A2C, DQN


@pytest.fixture
def algo(request):
    buffer_size = 10000
    algo_name = request.param
    if algo_name == "ppo":
        return sb3.PPO
    elif algo_name == "custom-ppo":
        return PatchedPPO
    elif algo_name == "sac":
        return lambda *args, **kwargs: sb3.SAC(*args, **kwargs, buffer_size=buffer_size)
    elif algo_name == "td3":
        return lambda *args, **kwargs: sb3.TD3(*args, **kwargs, buffer_size=buffer_size)
    elif algo_name == "ddpg":
        return lambda *args, **kwargs: sb3.DDPG(
            *args, **kwargs, buffer_size=buffer_size
        )
    elif algo_name == "a2c":
        return lambda *args, **kwargs: sb3.A2C(*args, **kwargs)
    elif algo_name == "dqn":
        return lambda *args, **kwargs: sb3.DQN(*args, **kwargs, buffer_size=buffer_size)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.env_util import make_vec_env


# Test exporting SB3 policies to ONNX, tests with every algorithm from SB3
@pytest.mark.parametrize(
    "env_class,algo",
    [
        ("discrete", "ppo"),
        ("continuous", "ppo"),
        ("continuous", "sac"),
        ("continuous", "td3"),
        ("continuous", "ddpg"),
        ("continuous", "a2c"),
        ("discrete", "dqn"),
    ],
    indirect=True,
)
def test_export_sb3_policy_to_onnx(tmp_path, env_class, algo):

    # Create a dummy environment
    env = make_vec_env(env_class, 2, seed=1)  # prevents an error with automatic seeding
    env = VecMergeDictActionWrapper(env)
    # Create a dummy model
    model: BaseAlgorithm = algo("MultiInputPolicy", env, verbose=1)
    # Train the model
    model.__original_action_space = env.unwrapped.action_space

    save_model_as_onnx(model, tmp_path / "test.onnx")


from schola.sb3.action_space_patch import ActionSpacePatch


@pytest.mark.parametrize("env_class", ["discrete", "continuous", "both"], indirect=True)
def test_export_custom_sb3_policy_to_onnx(tmp_path, env_class):
    # Create a dummy environment
    env = make_vec_env(env_class, 2, seed=1)  # prevents an error with automatic seeding

    # Create a dummy model
    with ActionSpacePatch(globals()):
        model: BaseAlgorithm = PPO("MultiInputPolicy", env, verbose=1)
        # Train the model
        model.__original_action_space = env.unwrapped.action_space

        save_model_as_onnx(model, tmp_path / "test.onnx")
