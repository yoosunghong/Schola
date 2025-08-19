# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy, MultiInputPolicy
from typing import Any, Dict, Optional, Tuple, Type
from imitation.algorithms import bc
from imitation.data import rollout, types
from dataclasses import asdict, fields
from schola.sb3.env import VecEnv
import gymnasium as gym
from schola.sb3.action_space_patch import PatchedPPO
from schola.sb3.action_space_patch import ActionSpacePatch
from schola.core.error_manager import ScholaErrorContextManager
from schola.sb3.utils import (
    SB3BCModel,
)
from schola.imitation.sb3.trajectory_utils import (
    read_expert_from_json,
    parse_space_definitions,
)
from argparse import ArgumentParser
from numpy import array, int64
import traceback
from schola.scripts.sb3.settings import SB3ScriptArgs
from schola.scripts.common import (
    ActivationFunctionEnum,
    add_unreal_process_args,
    add_checkpoint_args,
)


def make_parser() -> ArgumentParser:
    """
    Create an argument parser for launching training with stable baselines 3.

    Returns
    -------
    ArgumentParser
        The argument parser for the script.
    """
    parser = ArgumentParser(prog="Launch Schola Imitation Learning with SB3")

    add_unreal_process_args(parser)

    parser.add_argument(
        "--disable-eval",
        default=False,
        action="store_true",
        help="Disable evaluation of the model after training. Useful for short runs that might otherwise hang with an untrained model.",
    )

    add_checkpoint_args(parser)

    behaviour_group = parser.add_argument_group("Behavior Cloning Arguments")
    behaviour_group.add_argument(
        "--expert-path",
        type=str,
        default="",
        help="Path to the expert data file.",
    )
    behaviour_group.add_argument(
        "--cloning-epochs",
        type=int,
        default=10,
        help="Number of epochs to train the behavior cloning model.",
    )
    behaviour_group.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training the behavior cloning model.",
    )
    behaviour_group.add_argument(
        "--minibatch-size",
        type=int,
        default=64,
        help="Minibatch size for training the behavior cloning model.",
    )
    behaviour_group.add_argument(
        "--learning-rate",
        type=float,
        default=0.0003,
        help="Learning rate for the behavior cloning model.",
    )

    logging_group = parser.add_argument_group("Logging Arguments")
    logging_group.add_argument(
        "-scholav",
        "--schola-verbosity",
        type=int,
        default=0,
        help="Verbosity level for Schola environment logs.",
    )
    logging_group.add_argument(
        "-sb3v",
        "--sb3-verbosity",
        type=int,
        default=1,
        help="Verbosity level for Stable Baselines3 logs.",
    )

    architecture_group = parser.add_argument_group("Network Architecture Arguments")

    architecture_group.add_argument(
        "--policy-parameters",
        nargs="*",
        type=int,
        default=None,
        help="Network architecture for the policy",
    )
    architecture_group.add_argument(
        "--activation",
        type=ActivationFunctionEnum,
        default=ActivationFunctionEnum.ReLU,
        help="Activation function to use for the network",
    )

    return parser


def main(args: SB3ScriptArgs) -> Optional[Tuple[float, float]]:
    rng = np.random.default_rng()

    # TODO: Add rewards to the expert data
    obs, acts, rews = read_expert_from_json(args.expert_path)
    observation_space, action_space = parse_space_definitions(args.expert_path)

    with ScholaErrorContextManager() as err_ctxt, ActionSpacePatch(
        globals()
    ) as action_patch:

        trajectories = []
        for i, episode in enumerate(obs):
            trajectory = types.TrajectoryWithRew(
                obs=types.DictObs(
                    _d=obs[i],
                ),
                acts=array(acts[i], dtype=float),
                infos=None,
                terminal=True,
                rews=array(rews[i], dtype=float),
            )
            trajectories.append(trajectory)
        transitions = rollout.flatten_trajectories(trajectories)

    def lr_schedule(progress_remaining: float) -> float:
        """Learning rate schedule."""
        return args.learning_rate * progress_remaining

    policy_constructor = (
        MultiInputPolicy
        if isinstance(observation_space, gym.spaces.Dict)
        else MlpPolicy
    )

    policy = policy_constructor(
        observation_space,
        action_space,
        net_arch=args.policy_parameters,
        lr_schedule=lr_schedule,
    )

    bc_trainer = bc.BC(
        observation_space=observation_space,
        action_space=action_space,
        demonstrations=transitions,
        rng=rng,
        batch_size=args.batch_size,
        minibatch_size=args.minibatch_size,
        policy=policy,
        device="cpu",
    )

    print("Training a policy using Behavior Cloning")
    bc_trainer.train(n_epochs=args.cloning_epochs)

    if args.export_onnx:
        print("Converting to ONNX")
        model = SB3BCModel(
            bc_trainer.policy,
            action_space,
        ).to("cpu")
        model.save_as_onnx(args.checkpoint_dir + "/bc_final.onnx")

    if args.save_final_policy:
        bc_trainer.policy.save(args.checkpoint_dir + "/bc_final.zip")

    if not args.disable_eval and args.port is not None:
        print("...evaluating the model")
        env = VecEnv(args.make_unreal_connection(), verbosity=args.schola_verbosity)
        mean_reward, std_reward = evaluate_policy(
            bc_trainer.policy, env, n_eval_episodes=10, deterministic=True
        )
        print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
        env.close()
        return mean_reward, std_reward


def get_dataclass_args(args: Dict[str, Any], dataclass: Type[Any]) -> Dict[str, Any]:
    """
    Get the arguments for a dataclass from a dictionary, potentially containing additional arguments.

    Parameters
    ----------
    args : Dict[str,Any]
        The dictionary of arguments.

    dataclass : Type[Any]
        The dataclass to get the arguments for.

    Returns
    -------
    Dict[str,Any]
        The arguments for the dataclass.
    """
    return {k: v for k, v in args.items() if k in {f.name for f in fields(dataclass)}}


def main_from_cli() -> Optional[Tuple[float, float]]:
    """
    Main function for launching training with stable baselines 3 from the command line.

    Returns
    -------
    Optional[Tuple[float,float]]
        The mean and standard deviation of the rewards if evaluation is enabled, otherwise None.

    See Also
    --------
    main : The main function for launching training with stable baselines 3
    """
    parser = make_parser()

    args = parser.parse_args()
    args_dict = vars(args)
    sb3_args = get_dataclass_args(args_dict, SB3ScriptArgs)
    args = SB3ScriptArgs(**sb3_args)
    return main(args)


def debug_main_from_cli() -> None:
    """
    Debug main function for launching training with stable baselines 3 from the command line.

    See Also
    --------
    main_from_cli : The main function for launching training with stable baselines 3 from the command line
    main : The main function for launching training with stable baselines 3
    """
    try:
        main_from_cli()
    except Exception as e:
        traceback.print_exc()
    finally:
        input("Press any key to close:")


if __name__ == "__main__":
    debug_main_from_cli()
