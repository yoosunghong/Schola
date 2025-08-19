# Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Implementation of ray.rllib.env.base_env.BaseEnv backed by a Schola Environment.
"""

from typing import Any, List, Optional, Tuple, Dict, Union
import logging

from schola.core.unreal_connections import BaseUnrealConnection
from schola.core.env import AutoResetType, ScholaEnv, EnvAgentIdDict
from schola.core.spaces import (
    DictSpace,
)

from ray.rllib.env.base_env import BaseEnv as RayBaseEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.annotations import PublicAPI

logger = logging.getLogger(__name__)


def sorted_multi_agent_space(multi_agent_space: Dict[int, DictSpace]) -> DictSpace:
    """
    Sorts the spaces in a multi-agent space alphabetically by agent ID.

    Parameters
    ----------
    multi_agent_space : Dict[int,DictSpace]
        The multi-agent space to sort.

    Returns
    -------
    DictSpace
        The sorted multi-agent space.
    """
    output_space = DictSpace()
    for agent_id, original_space in multi_agent_space.items():
        sorted_space = DictSpace()
        for key in sorted(original_space):
            sorted_space[key] = original_space[key]
        output_space[agent_id] = sorted_space
    return output_space


@PublicAPI
class BaseEnv(RayBaseEnv):
    """
    A Ray RLlib environment that wraps a Schola environment.

    Parameters
    ----------
    unreal_connection : BaseUnrealConnection
        The connection to the Unreal Engine environment.
    verbosity : int, default=0
        The verbosity level for the environment.

    Attributes
    ----------
    unwrapped : MultiAgentEnv
        The underlying multi-agent environment.
    last_reset_obs : Dict[int,Dict[str,Any]]
        The observations recorded during the last reset.
    last_reset_infos : Dict[int,Dict[str,str]]
        The info dict recorded during the last reset.
    """

    def __init__(
        self,
        unreal_connection: BaseUnrealConnection,
        verbosity: int = 0,
    ):
        self.first_poll = True

        self._env = ScholaEnv(
            unreal_connection, verbosity, auto_reset_type=AutoResetType.SAME_STEP
        )
        self.last_reset_obs = {}
        self.last_reset_infos = {}

        class MultiAgentSubclass(MultiAgentEnv):
            def __init__(self, action_space, observation_space, agent_ids=None):
                self.action_space = action_space
                self.observation_space = observation_space
                self._obs_space_in_preferred_format = True
                self._action_space_in_preferred_format = True
                self._agent_ids = agent_ids
                super().__init__()

            def reset(self):
                pass

            def step(self, action_dict):
                pass

        # Use the first environment's action and observation space to create a mock MultiAgentEnv subclass
        # We can do this the parallel environments are homogeneous

        # Because of some oddity with Rllib, it parses the spaces in alphabetical order, so the space
        # definition must match. ~ Tian, Aug 2024
        observation_space = sorted_multi_agent_space(self._env.obs_defns[0])
        action_space = sorted_multi_agent_space(self._env.action_defns[0])

        logging.debug(action_space)
        logging.debug(observation_space)
        # we convert the dictionary to a Dict space
        self.unwrapped: MultiAgentEnv = MultiAgentSubclass(
            action_space=action_space,
            observation_space=observation_space,
            agent_ids=set(observation_space.keys()),
        )

    @property
    def observation_space(self) -> DictSpace:  # DictSpace[int,DictSpace[str,Any]]
        """
        The observation space for the environment.

        Returns
        -------
        DictSpace
            The observation space for the environment.
        """
        return self.unwrapped.observation_space

    @property
    def action_space(self) -> DictSpace:  # DictSpace[int,DictSpace[str,Any]]
        """
        The action space for the environment.

        Returns
        -------
        DictSpace
            The action space for the environment
        """
        return self.unwrapped.action_space

    @property
    def num_envs(self) -> int:
        """
        The number of sub-environments in the wrapped environment.

        Returns
        -------
        int
            The number of sub-environments in the wrapped environment.
        """
        return self._env.num_envs

    def poll(
        self,
    ) -> Tuple[
        EnvAgentIdDict[Dict[str, Any]],
        EnvAgentIdDict[float],
        EnvAgentIdDict[bool],
        EnvAgentIdDict[bool],
        EnvAgentIdDict[Dict[str, str]],
        EnvAgentIdDict[Any],
    ]:
        """
        Poll the environment for the next observation, reward, termination, info and any off_policy_actions (Currently Unused).

        Returns
        -------
        observations : EnvAgentIdDict[Dict[str,Any]]
            A dictionary, keyed by the environment and agent Id, containing the observations for each agent.
        rewards : EnvAgentIdDict[float]
            A dictionary, keyed by the environment and agent Id, containing the reward for each agent.
        terminateds : EnvAgentIdDict[bool]
            A dictionary, keyed by the environment and agent Id, containing the termination flag for each agent.
        truncateds : EnvAgentIdDict[bool]
            A dictionary, keyed by the environment and agent Id, containing the truncation flag for each agent.
        infos : EnvAgentIdDict[Dict[str,str]]]
            A dictionary, keyed by the environment and agent Id, containing the information dictionary for each agent.
        off_policy_actions : EnvAgentIdDict[Any]
            A dictionary, keyed by the environment and agent Id, containing the off-policy actions for each agent. Unused.
        """
        if self.first_poll:
            self.first_poll = False
            obs, rewards, terminateds, truncateds, infos = {}, {}, {}, {}, {}
            for env_id in self._env.obs_defns:
                rewards[env_id] = {}
                terminateds[env_id] = {}
                truncateds[env_id] = {}
                for agent_id in self._env.obs_defns[env_id]:
                    rewards[env_id][agent_id] = 0
                    terminateds[env_id][agent_id] = False
                    truncateds[env_id][agent_id] = False
            obs, infos = self._env.hard_reset()
        else:
            obs, rewards, terminateds, truncateds, infos = self._env.poll()

        off_policy_actions = {}  # TODO: Implement off-policy actions

        completed_env_ids = []
        for env_id in obs:
            terminateds[env_id]["__all__"] = all(terminateds[env_id].values())
            truncateds[env_id]["__all__"] = all(truncateds[env_id].values())
            if terminateds[env_id]["__all__"] or truncateds[env_id]["__all__"]:
                completed_env_ids.append(env_id)

        if completed_env_ids:
            self.last_reset_obs, self.last_reset_infos = self._env.soft_reset(
                completed_env_ids
            )

        # logging.info(f"{obs}, {terminateds},{truncateds}, {infos}")
        logging.info(f"{terminateds}")
        return obs, rewards, terminateds, truncateds, infos, off_policy_actions

    def send_actions(self, action_dict: EnvAgentIdDict[Dict[str, Any]]) -> None:
        self._env.send_actions(action_dict)

    def try_reset(
        self,
        env_id: Optional[int] = None,
        seed: Optional[Union[List[int], int]] = None,
        options: Optional[Dict[str, str]] = None,
    ) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, str]]]:
        logging.info(env_id)
        if env_id is not None:
            obs = {env_id: self.last_reset_obs[env_id]}
            infos = {env_id: self.last_reset_infos[env_id]}
            return obs, infos
        else:
            return self.last_reset_obs, self.last_reset_infos

    def stop(self) -> None:
        self._env.close()
