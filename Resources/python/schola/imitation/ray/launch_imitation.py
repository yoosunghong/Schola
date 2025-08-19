# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

from ray.rllib.algorithms.bc import BCConfig
from ray import air, tune
from ray.tune.registry import register_env
import ray
from ray.rllib.policy.policy import Policy
from schola.ray.utils import export_onnx_from_policy
import traceback
from argparse import ArgumentParser
from schola.gym.env import GymEnv
from typing import Any, Dict, Type
from dataclasses import fields
from schola.imitation.ray.convert_trajectories import convert_to_rllib_format
from schola.scripts.common import (
    add_unreal_process_args,
    add_checkpoint_args,
)
from schola.scripts.ray.settings import (
    TrainingSettings,
    ResumeSettings,
    LoggingSettings,
    NetworkArchitectureSettings,
    ResourceSettings,
    BehaviourCloningSettings,
    RLlibScriptArgs,
)


def make_parser() -> ArgumentParser:
    """
    Create an argument parser for launching training with stable baselines 3.

    Returns
    -------
    ArgumentParser
        The argument parser for the script.
    """
    parser = ArgumentParser(prog="Launch imitation training with Ray RLlib")
    parser = ArgumentParser(prog="Launch Schola Examples with RLlib")

    add_unreal_process_args(parser)

    training_args_group = parser.add_argument_group("Training Arguments")
    TrainingSettings.populate_arg_group(training_args_group)

    logging_args_group = parser.add_argument_group("Logging Arguments")
    LoggingSettings.populate_arg_group(logging_args_group)

    checkpoint_group = add_checkpoint_args(parser)

    ResumeSettings.populate_arg_group(checkpoint_group)

    architecture_group = parser.add_argument_group("Network Architecture Arguments")
    NetworkArchitectureSettings.populate_arg_group(architecture_group)

    resource_group = parser.add_argument_group("Resource Arguments")
    ResourceSettings.populate_arg_group(resource_group)

    behaviour_cloning_group = parser.add_argument_group("Behaviour Cloning Arguments")
    BehaviourCloningSettings.populate_arg_group(behaviour_cloning_group)

    return parser


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


def main_from_cli() -> None:
    """
    Main function for launching behaviour cloning with ray from the command line.

    See Also
    --------
    main : The main function for launching behaviour cloning with ray
    """
    parser = make_parser()
    args = parser.parse_args()
    args_dict = vars(args)

    # split the arguments into individual dictionaries for each dataclass
    training_args = get_dataclass_args(args_dict, TrainingSettings)
    logging_args = get_dataclass_args(args_dict, LoggingSettings)
    resume_args = get_dataclass_args(args_dict, ResumeSettings)
    network_args = get_dataclass_args(args_dict, NetworkArchitectureSettings)
    resource_args = get_dataclass_args(args_dict, ResourceSettings)
    behaviour_cloning_args = get_dataclass_args(args_dict, BehaviourCloningSettings)
    rllib_args = get_dataclass_args(args_dict, RLlibScriptArgs)

    # build datraclasses from the dictionaries
    training_args = TrainingSettings(**training_args)
    logging_args = LoggingSettings(**logging_args)
    resume_args = ResumeSettings(**resume_args)
    network_args = NetworkArchitectureSettings(**network_args)
    resource_args = ResourceSettings(**resource_args)
    behaviour_cloning_args = BehaviourCloningSettings(**behaviour_cloning_args)

    args = RLlibScriptArgs(
        training_settings=training_args,
        logging_settings=logging_args,
        network_architecture_settings=network_args,
        resource_settings=resource_args,
        behaviour_cloning_settings=behaviour_cloning_args,
        **rllib_args
    )

    return main(args)


def main(args: RLlibScriptArgs) -> None:

    rllib_expert_path = convert_to_rllib_format(
        args.behaviour_cloning_settings.expert_path,
        args.behaviour_cloning_settings.converted_expert_path,
    )

    config = BCConfig()
    config.training(
        lr=args.training_settings.learning_rate,
        model={
            "fcnet_hiddens": args.network_architecture_settings.fcnet_hiddens,
            "fcnet_activation": args.network_architecture_settings.activation.layer,
        },
    )
    config.offline_data(
        input_=[rllib_expert_path],
        actions_in_input_normalized=True,
    )
    config.resources(
        num_gpus=args.resource_settings.num_gpus,
        num_cpus_per_worker=args.resource_settings.num_cpus_for_main_process,
    )
    config.learners(
        num_learners=args.resource_settings.num_learners,
        num_cpus_per_learner=args.resource_settings.num_cpus_per_learner,
        num_gpus_per_learner=args.resource_settings.num_gpus_per_learner,
    )

    def env_creator(env_config):
        env = GymEnv(
            args.make_unreal_connection(),
            args.logging_settings.schola_verbosity,
        )
        return env

    register_env("schola_env", env_creator)

    config.environment(env="schola_env")

    stop = {
        "timesteps_total": args.behaviour_cloning_settings.cloning_steps,
    }

    ray.init()
    print("Starting training")
    try:
        results = tune.run(
            "BC",
            config=config,
            stop=stop,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_at_end=True,
            ),
            storage_path=args.checkpoint_dir,
        )
        last_checkpoint = results.get_last_checkpoint()

        print("Training complete")
    finally:
        # Always shutdown ray and release the environment from training even if there is an error
        # will reraise the error unless a control flow statement is added
        ray.shutdown()

    if args.export_onnx:
        export_onnx_from_policy(
            Policy.from_checkpoint(last_checkpoint), results.trials[-1].path
        )
        print("Models exported to ONNX at ", results.trials[-1].path)


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
