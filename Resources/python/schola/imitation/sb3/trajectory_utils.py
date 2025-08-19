# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

import numpy as np
import json
from schola.core.spaces.dict import DictSpace
from schola.core.spaces.box import BoxSpace
from schola.core.spaces.discrete import DiscreteSpace, MultiDiscreteSpace
from schola.core.spaces.binary import MultiBinarySpace


def read_expert_from_json(expert_path: str):

    # read the expert transitions from json
    obs = []
    acts = []
    rews = []
    num_eps = 0

    with open(expert_path, "r") as f:
        data = json.load(f)
        data = data["trajectory"]["steps"]
        for i, step in enumerate(data):
            # If this is the first step of the episode
            if step["stepNum"] == 0:
                # Start a new episode
                obs.append({})
                acts.append([])
                rews.append([])
                num_eps += 1
            step_obs = step["observations"]
            step_act = step["actions"]
            step_rews = step["reward"] if "reward" in step else 0
            for sensor_data in step_obs:
                sensor_name = sensor_data["interactorName"]
                sensor_value = [float(data) for data in sensor_data["value"].split(",")]
                if sensor_name not in obs[-1]:
                    obs[-1][sensor_name] = []
                obs[-1][sensor_name].append([sensor_value])
            step_act_concat = []
            for actuator_data in step_act:
                actuator_name = actuator_data["interactorName"]
                actuator_value = [
                    float(data) for data in actuator_data["value"].split(",")
                ]
                step_act_concat.extend(actuator_value)
            acts[-1].append(step_act_concat)
            rews[-1].append(step_rews)

            # If this is the last step of the episode
            if i == len(data) - 1 or data[i + 1]["stepNum"] == 0:
                # Drop the last action and reward since the last step was terminal
                # Convert last episode's observation to numpy array
                acts[-1].pop()
                rews[-1].pop()
                for key in obs[-1]:
                    obs[-1][key] = np.array(obs[-1][key], dtype=float)
    return obs, acts, rews


def parse_space_definitions(expert_path: str):
    """
    Parse the observation space from a dictionary to a format suitable for gym spaces.

    Parameters
    ----------
    observation : dict
        The observation space as a dictionary.

    Returns
    -------
    dict
        The parsed observation space.
    """
    observation_space_dict = {}
    action_space_dict = {}

    data_path = expert_path
    with open(data_path, "r") as f:
        data = json.load(f)
        for sensor in data["discreteObservationSpaces"]:
            sensor_name = sensor
            sensor_high = data["discreteObservationSpaces"][sensor]["high"]
            observation_definition = (
                DiscreteSpace(n=sensor_high[0])
                if len(sensor_high) == 1
                else MultiDiscreteSpace(nvec=sensor_high)
            )
            observation_space_dict[sensor_name] = observation_definition

        for sensor in data["binaryObservationSpaces"]:
            sensor_name = sensor
            sensor_shape = data["binaryObservationSpaces"][sensor]["shape"]
            observation_definition = MultiBinarySpace(n=sensor_shape)
            observation_space_dict[sensor_name] = observation_definition

        for sensor in data["boxObservationSpaces"]:
            sensor_name = sensor
            sensor_low = data["boxObservationSpaces"][sensor]["dimensions"][0]["low"]
            sensor_high = data["boxObservationSpaces"][sensor]["dimensions"][0]["high"]
            sensor_shape = (
                tuple(data["boxObservationSpaces"][sensor]["shape"])
                if len(data["boxObservationSpaces"][sensor]["shape"]) > 0
                else (len(data["boxObservationSpaces"][sensor]["dimensions"]),)
            )
            observation_definition = BoxSpace(
                low=sensor_low, high=sensor_high, shape=sensor_shape
            )
            observation_space_dict[sensor_name] = observation_definition

    for actuator in data["discreteActionSpaces"]:
        actuator_name = actuator
        actuator_high = data["discreteActionSpaces"][actuator]["high"]
        action_definition = (
            DiscreteSpace(n=actuator_high[0])
            if len(actuator_high) == 1
            else MultiDiscreteSpace(nvec=actuator_high)
        )
        action_space_dict[actuator_name] = action_definition
    for actuator in data["binaryActionSpaces"]:
        actuator_name = actuator
        actuator_shape = data["binaryActionSpaces"][actuator]["shape"]
        action_definition = MultiBinarySpace(n=actuator_shape)
        action_space_dict[actuator_name] = action_definition
    for actuator in data["boxActionSpaces"]:
        actuator_name = actuator
        actuator_low = data["boxActionSpaces"][actuator]["dimensions"][0]["low"]
        actuator_high = data["boxActionSpaces"][actuator]["dimensions"][0]["high"]
        actuator_shape = (
            tuple(data["boxActionSpaces"][actuator]["shape"])
            if len(data["boxActionSpaces"][actuator]["shape"]) > 0
            else (len(data["boxActionSpaces"][actuator]["dimensions"]),)
        )
        action_definition = BoxSpace(
            low=actuator_low, high=actuator_high, shape=actuator_shape
        )
        action_space_dict[actuator_name] = action_definition

    observation_space = DictSpace(observation_space_dict)
    action_space = DictSpace(action_space_dict)

    return observation_space, action_space
