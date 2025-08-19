# Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Implementation of stable_baselines3.common.vec_env.VecEnv backed by a Schola Environment.
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, TypeVar, Union
from schola.core.env import AutoResetType, ScholaEnv
from schola.core.unreal_connections.base_connection import BaseUnrealConnection
from stable_baselines3.common.vec_env import VecEnv as Sb3VecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import _flatten_obs

import numpy as np
from schola.core.error_manager import EnvironmentException
from schola.core.utils.id_manager import nested_get, IdManager
import logging

T = TypeVar("T")


class VecEnv(Sb3VecEnv):
    def __init__(self, unreal_connection: BaseUnrealConnection, verbosity: int = 0):
        self._env = ScholaEnv(
            unreal_connection,
            verbosity,
            auto_reset_type=AutoResetType.SAME_STEP,
        )
        self.id_manager = IdManager(self._env.ids)
        # we just use the default UID to get the shared definition
        obs_space = self._env.get_obs_space(*self.id_manager[0])
        action_space = self._env.get_action_space(*self.id_manager[0])

        # test that everything is setup correctly
        try:
            for env_id, agent_id in self.id_manager.id_list:
                assert (
                    self._env.get_action_space(env_id, agent_id) == action_space
                ), f"Action Space Mismatch on Agent:{agent_id} in Env {env_id}.\nGot: {self._env.get_action_space(env_id,agent_id)}\nExpected:{action_space}"
                assert (
                    self._env.get_obs_space(env_id, agent_id) == obs_space
                ), f"Observation Space Mismatch on Agent:{agent_id} in Env {env_id}.\nGot: {self._env.get_obs_space(env_id,agent_id)}\nExpected:{obs_space}"
        except Exception as e:
            self._env.close()
            raise e
        logging.debug(action_space)
        logging.debug(obs_space)
        self.reset_infos = [{} for _ in range(self._env.num_agents)]
        self._seed: Optional[int] = None
        self.options: Optional[Dict[str, str]] = None
        super().__init__(self._env.num_agents, obs_space, action_space)

    def close(self) -> None:
        return self._env.close()

    def env_method(method_name, *method_args, indices=None, **method_kwargs): ...

    def get_attr(self, attr_name, indices=None):
        return [None for x in range(0, self._env.num_envs)]

    def reset(self) -> Dict[str, np.ndarray]:
        obs, nested_infos = self._env.hard_reset(seeds=self._seed, options=self.options)

        self._seed = None
        self.options = None

        for env_id in nested_infos:
            for agent_id in nested_infos[env_id]:
                uid = self.id_manager[env_id, agent_id]
                self.reset_infos[uid] = nested_infos[env_id][agent_id]
        # flatten the observations, converting from dict to list using key as indices
        obs = self.id_manager.flatten_id_dict(obs)
        # flatten even more, for sb3 compatibility
        obs = _flatten_obs(obs, self.observation_space)
        return obs

    def env_is_wrapped(self, wrapper_class, indices=None) -> bool:
        if indices is None:
            indices = (x for x in range(len(self._env.ids)))
        return [False for x in indices]

    def seed(self, seed: Optional[int] = None) -> None:
        if not seed is None:
            self._seed = seed

    def set_options(self, options: Optional[Dict[str, str]] = None) -> None:
        """
        Set the options for the environment.

        Parameters
        ----------
        options : Optional[Dict[str,str]], optional
            The options to set, by default None.
        """
        if not options is None:
            self.options = options

    def set_attr(self, attr_name, value, indices=None): ...

    def step_async(
        self, actions: Union[List[np.ndarray], List[Dict[str, np.ndarray]]]
    ) -> None:
        # actions can come in as a list of flattened tensors so we need to unflatten them
        if isinstance(actions[0], np.ndarray):
            unflattened_actions = [OrderedDict() for _ in actions]
            for i, action in enumerate(actions):
                start_dim = 0
                for name, space in self.action_space.items():
                    # TODO apply any necessary DTYPE conversions here (everything is a float32 otherwise)
                    unflattened_actions[i][name] = action[
                        start_dim : start_dim + len(space)
                    ]
                    start_dim += len(space)
        else:
            # actions came in as a dict. how nice!
            unflattened_actions = actions
        # convert vector to Nested dictionary
        actions = self.id_manager.nest_id_list(unflattened_actions)

        self._env.send_actions(actions)

    def step_wait(
        self,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, List[Dict[str, str]]]:
        observations, rewards, terminateds, truncateds, nested_infos = self._env.poll()

        array_dones = np.empty((self._env.num_agents,), dtype=np.bool_)

        array_rewards = np.asarray(self.id_manager.flatten_id_dict(rewards))

        array_observations = self.id_manager.flatten_id_dict(observations)

        infos = [{} for _ in range(self._env.num_agents)]
        for env_id in nested_infos:
            for agent_id in nested_infos[env_id]:
                uid = self.id_manager[env_id, agent_id]
                # safe because we are iterating over nested_infos
                infos[uid] = nested_infos[env_id][agent_id]

        envs_to_reset = []

        for env_id, agent_id_list in enumerate(self.id_manager.ids):
            any_done = False
            all_done = True
            for agent_id in agent_id_list:
                uid = self.id_manager[env_id, agent_id]
                array_dones[uid] = nested_get(
                    truncateds, (env_id, agent_id), False
                ) or nested_get(terminateds, (env_id, agent_id), False)
                any_done = any_done or array_dones[uid]
                all_done = all_done and array_dones[uid]

            # We don't handle the case where 1 agent ends early currently.
            if any_done:
                if all_done:
                    envs_to_reset.append(env_id)
                else:
                    raise EnvironmentException(
                        f"SB3 with multi-agent environments does not support agents completing at different steps. Env {env_id} had agents in different completion states."
                    )

        # following the sb3 vec env guideline we self reset
        if len(envs_to_reset) > 0:
            resetted_obs, reset_infos = self._env.soft_reset(envs_to_reset)
            self.reset_infos = [{} for _ in range(self._env.num_agents)]
            for env_id in reset_infos:
                for agent_id in reset_infos[env_id]:
                    uid = self.id_manager[env_id, agent_id]
                    # safe because we are iterating over nested_infos
                    self.reset_infos[uid] = reset_infos[env_id][agent_id]

            for env_id in envs_to_reset:
                for agent_id in self.id_manager.partial_get(env_id):
                    uid = self.id_manager[env_id, agent_id]
                    # Observations of the last step of the episode go into the info section
                    infos[uid]["terminal_observation"] = observations[env_id][agent_id]
                    infos[uid]["TimeLimit.truncated"] = (
                        truncateds[env_id][agent_id]
                        and not terminateds[env_id][agent_id]
                    )

                    # put the new observations from the start of the new episode into the returned obs
                    array_observations[uid] = resetted_obs[env_id][agent_id]

        return (
            _flatten_obs(array_observations, self.observation_space),
            array_rewards,
            array_dones,
            infos,
        )
