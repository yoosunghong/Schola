# Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Script to train an rllib model using Schola.
"""
from argparse import ArgumentParser
from schola.ray.utils import export_onnx_from_policy
from typing import Any, Dict, Type, Union
import traceback

from schola.ray.env import BaseEnv
from schola.ray.utils import MultiAgentTransposeImageWrapper
from schola.core.env import ScholaEnv
from schola.core.utils.plugins import get_plugins

import ray
from ray import air, tune
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.appo.appo import APPOConfig
from ray.rllib.algorithms.impala.impala import IMPALAConfig
from ray.tune.registry import register_env
from schola.scripts.common import (
    add_unreal_process_args,
    add_checkpoint_args,
)
from dataclasses import fields
from ray.rllib.policy.policy import Policy
from schola.scripts.ray.settings import (
    TrainingSettings,
    ResumeSettings,
    LoggingSettings,
    NetworkArchitectureSettings,
    ResourceSettings,
    RLlibScriptArgs,
    PPOSettings,
    APPOSettings,
    IMPALASettings,
)


def make_parser():
    """
    Create an argument parser for launching training with ray.

    Returns
    -------
    ArgumentParser
        The argument parser for the script.
    """
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

    subparsers = parser.add_subparsers(
        required=True, help="Choose the algorithm to use"
    )

    ppo_parser = subparsers.add_parser(
        "PPO", help="Proximal Policy Optimization", parents=[PPOSettings.get_parser()]
    )
    appo_parser = subparsers.add_parser(
        "APPO",
        help="Asynchronous Proximal Policy Optimization",
        parents=[APPOSettings.get_parser()],
    )
    impala_parser = subparsers.add_parser(
        "IMPALA",
        help="Importance Weighted Actor-Learner Architecture",
        parents=[IMPALASettings.get_parser()],
    )

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


def main_from_cli() -> tune.ExperimentAnalysis:
    """
    Main function for launching training with ray from the command line.

    Returns
    -------
    tune.ExperimentAnalysis
        The results of the training

    See Also
    --------
    main : The main function for launching training with ray
    """
    parser = make_parser()

    discovered_plugins = get_plugins("schola.plugins.ray.launch")
    for plugin in discovered_plugins:
        plugin.add_plugin_args_to_parser(parser)

    args = parser.parse_args()
    args_dict = vars(args)
    # split the arguments into individual dictionaries for each dataclass
    algorithm_args = get_dataclass_args(args_dict, args.algorithm_settings_class)
    training_args = get_dataclass_args(args_dict, TrainingSettings)
    logging_args = get_dataclass_args(args_dict, LoggingSettings)
    resume_args = get_dataclass_args(args_dict, ResumeSettings)
    network_args = get_dataclass_args(args_dict, NetworkArchitectureSettings)
    resource_args = get_dataclass_args(args_dict, ResourceSettings)
    rllib_args = get_dataclass_args(args_dict, RLlibScriptArgs)

    # build datraclasses from the dictionaries
    algorithm_args = args.algorithm_settings_class(**algorithm_args)
    training_args = TrainingSettings(**training_args)
    logging_args = LoggingSettings(**logging_args)
    resume_args = ResumeSettings(**resume_args)
    network_args = NetworkArchitectureSettings(**network_args)
    resource_args = ResourceSettings(**resource_args)

    plugins = []
    for plugin in discovered_plugins:
        plugin_args = get_dataclass_args(args_dict, plugin)
        plugins.append(plugin(**plugin_args))

    args = RLlibScriptArgs(
        algorithm_settings=algorithm_args,
        training_settings=training_args,
        logging_settings=logging_args,
        resume_settings=resume_args,
        network_architecture_settings=network_args,
        resource_settings=resource_args,
        plugins=plugins,
        **rllib_args
    )

    return main(args)


def main(args: RLlibScriptArgs) -> tune.ExperimentAnalysis:
    """
    Main function for launching training with ray.

    Parameters
    ----------
    args : RLlibArgs
        The arguments for the script as a dataclass

    Returns
    -------
    tune.ExperimentAnalysis
        The results of the training
    """
    # collect the names of the agents by creating a temporary environment
    schola_env = ScholaEnv(
        args.make_unreal_connection(), verbosity=args.logging_settings.schola_verbosity
    )
    agent_names = schola_env.agent_display_names[0]
    schola_env.close()

    # Clusters configure resources automatically
    if args.resource_settings.using_cluster:
        ray.init()
    else:
        ray.init(
            num_cpus=args.resource_settings.num_cpus,
            num_gpus=args.resource_settings.num_gpus,
        )

    def env_creator(env_config):
        env = BaseEnv(
            args.make_unreal_connection(),
            verbosity=args.logging_settings.schola_verbosity,
        )
        env = MultiAgentTransposeImageWrapper(env)
        return env

    def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
        return agent_names[agent_id]

    register_env("schola_env", env_creator)
    # Note New Ray Stack doesn't support Vectorized MutiAgent environments yet so the old stack is better
    config: Union[PPOConfig, APPOConfig, IMPALAConfig] = (
        args.algorithm_settings.rllib_config()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            "schola_env", clip_rewards=False, clip_actions=True, normalize_actions=False
        )
        .framework("torch")
        .env_runners(num_env_runners=0, num_envs_per_env_runner=1)
        .multi_agent(
            policies={
                agent_name: PolicySpec(observation_space=None, action_space=None)
                for agent_name in set(agent_names.values())
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=None,  # default to training all policies
        )
        .resources(
            num_cpus_for_main_process=args.resource_settings.num_cpus_for_main_process,
            num_gpus=args.resource_settings.num_gpus,
        )
        .learners(
            num_learners=args.resource_settings.num_learners,
            num_cpus_per_learner=args.resource_settings.num_cpus_per_learner,
            num_gpus_per_learner=args.resource_settings.num_gpus_per_learner,
        )
        .training(
            lr=args.training_settings.learning_rate,
            gamma=args.training_settings.gamma,
            num_sgd_iter=args.training_settings.num_sgd_iter,
            train_batch_size_per_learner=args.training_settings.train_batch_size_per_learner,
            minibatch_size=args.training_settings.minibatch_size,
            model={
                "fcnet_hiddens": args.network_architecture_settings.fcnet_hiddens,
                "fcnet_activation": args.network_architecture_settings.activation.layer,
                "free_log_std": False,  # onnx fails to load if this is set to True
                "use_attention": args.network_architecture_settings.use_attention,
                "attention_dim": args.network_architecture_settings.attention_dim,
            },
            **args.algorithm_settings.get_settings_dict()
        )
    )

    stop = {
        "timesteps_total": args.training_settings.timesteps,
    }

    callbacks = []
    for plugin in args.plugins:
        callbacks += plugin.get_extra_callbacks()

    print("Starting training")
    try:
        results = tune.run(
            args.algorithm_settings.name,
            config=config,
            stop=stop,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=args.save_freq if args.enable_checkpoints else 0,
                checkpoint_at_end=args.save_final_policy,
            ),
            restore=args.resume_settings.resume_from,
            verbose=args.logging_settings.rllib_verbosity,
            storage_path=args.checkpoint_dir,
            callbacks=callbacks,
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
    return results


def debug_main_from_cli() -> None:
    """
    Main function for launching training with ray from the command line, that catches any errors and waits for user input to close.

    See Also
    --------
    main_from_cli : The main function for launching training with ray from the command line
    main : The main function for launching training with ray
    """
    try:
        main_from_cli()
    except Exception as e:
        traceback.print_exc()
    finally:
        input("Press any key to close:")


if __name__ == "__main__":
    debug_main_from_cli()
