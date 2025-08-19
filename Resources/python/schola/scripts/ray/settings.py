# Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
"""
This module contains the settings dataclasses for the RLlib script
"""
import argparse
from typing import Any, Dict, List, Optional, Type, Union
from dataclasses import dataclass, field
from ray.rllib.algorithms.appo.appo import APPOConfig
from ray.rllib.algorithms.impala.impala import IMPALAConfig
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from schola.scripts.common import (
    ActivationFunctionEnum,
    ScriptArgs,
    RLLibLauncherExtension,
)


class RLLibAlgorithmSpecificSettings:
    """
    Base Class for RLLib algorithm specific settings. This class is intended to be inherited by specific algorithm settings classes (e.g., PPOSettings, IMPALASettings, etc.).
    """

    def get_settings_dict(self) -> Dict[str, Any]:
        """
        Get the settings as a dictionary keyed by the correct parameter name in Ray
        """
        ...

    @classmethod
    def get_parser(cls):
        """
        Add the settings to the parser or subparser
        """
        ...


@dataclass
class PPOSettings(RLLibAlgorithmSpecificSettings):
    """
    Dataclass for PPO (Proximal Policy Optimization) algorithm specific settings. This class defines the parameters used in the PPO algorithm, including GAE lambda, clip parameter, and whether to use GAE.
    """

    gae_lambda: float = (
        0.95  #: The lambda parameter for Generalized Advantage Estimation (GAE). This controls the trade-off between bias and variance in the advantage estimation.
    )
    clip_param: float = (
        0.2  #: The clip parameter for the PPO algorithm. This is the epsilon value used in the clipped surrogate objective function. It helps to limit the policy update step size to prevent large changes that could lead to performance collapse.
    )
    use_gae: bool = (
        True  #: Whether to use Generalized Advantage Estimation (GAE) for advantage calculation. GAE is a method to reduce the variance of the advantage estimates while keeping bias low. If set to False, the standard advantage calculation will be used instead.
    )

    @property
    def rllib_config(self) -> Type[PPOConfig]:
        return PPOConfig

    @property
    def name(self) -> str:
        return "PPO"

    def get_settings_dict(self):
        return {
            "lambda_": self.gae_lambda,
            "use_gae": self.use_gae,
            "clip_param": self.clip_param,
        }

    @classmethod
    def get_parser(cls):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--disable-gae",
            dest="use_gae",
            action="store_false",
            help="Disable the Generalized Advantage Estimation (GAE) for the PPO algorithm",
        )
        parser.add_argument(
            "--gae-lambda",
            type=float,
            default=0.95,
            help="The GAE lambda value for the PPO algorithm",
        )
        parser.add_argument(
            "--clip-param",
            type=float,
            default=0.2,
            help="The clip range for the PPO algorithm",
        )
        parser.set_defaults(algorithm_settings_class=PPOSettings)
        return parser


@dataclass
class IMPALASettings(RLLibAlgorithmSpecificSettings):
    """
    Dataclass for IMPALA (Importance Weighted Actor-Learner Architecture) algorithm specific settings. This class defines the parameters used in the IMPALA algorithm, including V-trace settings for off-policy correction.
    """

    vtrace: bool = (
        True  #: Whether to use the V-trace algorithm for off-policy correction in the IMPALA algorithm. V-trace is a method to correct the bias introduced by using off-policy data for training. It helps to ensure that the value estimates are more accurate and stable.
    )
    vtrace_clip_rho_threshold: float = (
        1.0  #: The clip threshold for V-trace rho values.
    )
    vtrace_clip_pg_rho_threshold: float = (
        1.0  #: The clip threshold for V-trace rho values in the policy gradient.
    )

    @property
    def rllib_config(self) -> Type[IMPALAConfig]:
        return IMPALAConfig

    @property
    def name(self) -> str:
        return "IMPALA"

    def get_settings_dict(self):
        return {
            "vtrace": self.vtrace,
            "vtrace_clip_rho_threshold": self.vtrace_clip_rho_threshold,
            "vtrace_clip_pg_rho_threshold": self.vtrace_clip_pg_rho_threshold,
        }

    @classmethod
    def get_parser(cls):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--disable-vtrace",
            dest="vtrace",
            action="store_false",
            help="Disable the V-trace algorithm",
        )
        parser.add_argument(
            "--vtrace-clip-rho-threshold",
            type=float,
            default=1.0,
            help="The clip threshold for V-trace rho values",
        )
        parser.add_argument(
            "--vtrace-clip-pg-rho-threshold",
            type=float,
            default=1.0,
            help="The clip threshold for V-trace rho values in the policy gradient",
        )
        parser.set_defaults(algorithm_settings_class=IMPALASettings)
        return parser


@dataclass
class APPOSettings(IMPALASettings, PPOSettings):
    """
    Dataclass for APPO (Asynchronous Proximal Policy Optimization) algorithm specific settings. This class inherits from both IMPALASettings and PPOSettings to combine the settings for both algorithms. This allows for the use of both V-trace for off-policy correction and PPO for policy optimization in a single algorithm.
    """

    @property
    def rllib_config(self) -> Type[APPOConfig]:
        return APPOConfig

    @property
    def name(self) -> str:
        return "APPO"

    def get_settings_dict(self):
        base_imapala_dict = IMPALASettings.get_settings_dict(self)
        base_ppo_dict = PPOSettings.get_settings_dict(self)
        return {**base_imapala_dict, **base_ppo_dict}

    @classmethod
    def get_parser(cls):
        parser = argparse.ArgumentParser(
            parents=[IMPALASettings.get_parser(), PPOSettings.get_parser()],
            add_help=False,
        )
        parser.set_defaults(algorithm_settings_class=APPOSettings)
        return parser


@dataclass
class TrainingSettings:
    """
    Dataclass for generic training settings used in the RLlib training process. This class defines the parameters for training, including the number of timesteps, learning rate, minibatch size, and other hyperparameters that control the training process. These settings are applicable to any RLlib algorithm and can be customized based on the specific requirements of the training job.
    """

    timesteps: int = (
        3000  #: The number of timesteps to train for. This is the total number of timesteps to run during training.
    )
    learning_rate: float = (
        0.0003  #: The learning rate for  any chosen algorithm. This controls how much to adjust the model weights in response to the estimated error each time the model weights are updated. A smaller value means slower learning, while a larger value means faster learning.
    )
    minibatch_size: int = (
        128  #: The size of the minibatch for training. This is the number of samples used in each iteration of training to update the model weights. A larger batch size can lead to more stable estimates of the gradient, but requires more memory and can slow down training if too large.
    )
    train_batch_size_per_learner: int = (
        256  #: The number of samples given to each learner during training. Must be divisble by minibatch_size.
    )
    num_sgd_iter: int = (
        5  #: The number of stochastic gradient descent (SGD) iterations for each batch. This is the number of times to update the model weights using the samples in the minibatch. More iterations can lead to better convergence, but also increases the training time.
    )
    gamma: float = (
        0.99  #: The discount factor for the reinforcement learning algorithm. This is used to calculate the present value of future rewards. A value of 0.99 means that future rewards are discounted by 1% for each time step into the future. This helps to balance the importance of immediate versus future rewards in the training process. A value closer to 1.0 will prioritize future rewards more heavily, while a value closer to 0 will prioritize immediate rewards.
    )

    @property
    def name(self) -> str:
        return "Training Settings"

    @classmethod
    def populate_arg_group(cls, args_group):
        args_group.add_argument(
            "-t",
            "--timesteps",
            type=int,
            default=3000,
            help="Number of timesteps to train for",
        )
        args_group.add_argument(
            "--learning-rate",
            type=float,
            default=0.0003,
            help="Learning rate for the PPO algorithm",
        )
        args_group.add_argument(
            "--minibatch-size",
            type=int,
            default=128,
            help="The size of the minibatch for training. Taken from the train batch given to each learner",
        )
        args_group.add_argument(
            "--train-batch-size-per-learner",
            type=int,
            default=256,
            help="Size of the minibatch given to each learner",
        )
        args_group.add_argument(
            "--num-sgd-iter",
            type=int,
            default=5,
            help="The number of SGD iterations for each batch",
        )
        args_group.add_argument(
            "--gamma",
            type=float,
            default=0.99,
            help="The discount factor for the PPO algorithm",
        )


@dataclass
class ResourceSettings:
    """
    Dataclass for resource settings used in the RLlib training process. This class defines the parameters for allocating computational resources, including the number of GPUs and CPUs to use for the training job. These settings help to control how resources are allocated for the training process, which can impact performance and training times. This is especially important when running on a cluster or distributed environment.
    """

    num_gpus: Optional[int] = (
        0  #: The number of GPUs to use for the training process. This specifies how many GPUs are available for the RLlib training job. If set to 0, it will default to CPU training. This can be used to leverage GPU acceleration for faster training times if available.
    )
    num_cpus: Optional[int] = (
        1  #: The total number of CPUs to use for the training process. This specifies how many CPU cores are available for the RLlib training job. This can be used to parallelize the training process across multiple CPU cores, which can help to speed up training times.
    )
    num_learners: Optional[int] = (
        0  #: The number of learner processes to use for the training job. This specifies how many parallel learner processes will be used to train the model. Each learner will process a portion of the training data and update the model weights independently. This can help to speed up training times by leveraging multiple CPU cores or GPUs.
    )
    num_cpus_for_main_process: Optional[int] = (
        1  #: The number of CPUs to allocate for the main process. This is the number of CPU cores that will be allocated to the main process that manages the training job. This can be used to ensure that the main process has enough resources to handle the workload and manage the learner processes effectively.
    )
    num_cpus_per_learner: Optional[int] = (
        1  #: The number of CPUs to allocate for each learner process. This specifies how many CPU cores will be allocated to each individual learner process that is used for training. This can be used to ensure that each learner has enough resources to handle its workload and process the training data efficiently.
    )
    num_gpus_per_learner: Optional[int] = (
        0  #: The number of GPUs to allocate for each learner process. This specifies how many GPUs will be allocated to each individual learner process that is used for training.
    )
    using_cluster: Optional[bool] = (
        False  #: Whether Ray is running on a predefined cluster, or if one should be created as part of the launch script.
    )

    @property
    def name(self) -> str:
        return "Resource Settings"

    @classmethod
    def populate_arg_group(cls, args_group):
        args_group.add_argument(
            "--num-gpus", type=int, default=0, help="Number of GPUs to use"
        )
        args_group.add_argument(
            "--num-cpus", type=int, default=1, help="Number of CPUs to use"
        )
        args_group.add_argument(
            "--num-cpus-per-learner",
            type=int,
            default=1,
            help="Number of CPUs to use per learner process",
        )
        args_group.add_argument(
            "--num-gpus-per-learner",
            type=int,
            default=0,
            help="Number of GPUs to use per learner process",
        )
        args_group.add_argument(
            "--num-learners",
            type=int,
            default=0,
            help="Number of learner processes to use",
        )
        args_group.add_argument(
            "--num-cpus-for-main-process",
            type=int,
            default=1,
            help="Number of CPUs to use for the main process",
        )
        args_group.add_argument(
            "--using-cluster",
            action="store_true",
            help="Whether Ray is running on a cluster",
        )


@dataclass
class LoggingSettings:
    """
    Dataclass for logging settings used in the RLlib training process. This class defines the verbosity levels for logging in both the Schola environment and RLlib. These settings help to control the amount of logging information generated during the training process, which can be useful for debugging and understanding the training process. Adjusting these settings can help to balance the amount of information logged against performance and readability of the logs.
    """

    schola_verbosity: int = (
        0  #: Verbosity level for the Schola environment. This controls the level of detail in the logs generated by the Schola environment. A higher value will produce more detailed logs, which can be useful for debugging and understanding the training process. Default is 0 (no additional logging).
    )
    rllib_verbosity: int = (
        1  #: Verbosity level for RLlib. This controls the level of detail in the logs generated by RLlib. A higher value will produce more detailed logs, which can be useful for debugging and understanding the training process. Default is 1 (standard logging).
    )

    @property
    def name(self) -> str:
        return "Logging Settings"

    @classmethod
    def populate_arg_group(cls, args_group):
        args_group.add_argument(
            "-scholav",
            "--schola-verbosity",
            type=int,
            default=0,
            help="Verbosity level for the Schola environment",
        )
        args_group.add_argument(
            "-rllibv",
            "--rllib-verbosity",
            type=int,
            default=1,
            help="Verbosity level for RLlib",
        )


@dataclass
class ResumeSettings:
    """
    Dataclass for resume settings used in the RLlib training process. This class defines the parameters for resuming training from a saved checkpoint. This allows you to continue training from a previously saved model checkpoint instead of starting from scratch. This is useful for long training jobs or if you want to experiment with different hyperparameters without losing progress.
    """

    resume_from: Optional[str] = (
        None  #: Path to a checkpoint to resume training from. This allows you to continue training from a previously saved model checkpoint instead of starting from scratch. This is useful for long training jobs or if you want to experiment with different hyperparameters without losing progress. If set to None, training will start from scratch.
    )

    @property
    def name(self) -> str:
        return "Resume Settings"

    @classmethod
    def populate_arg_group(cls, args_group):
        args_group.add_argument(
            "--resume-from",
            type=str,
            default=None,
            help="Path to checkpoint to resume from",
        )


@dataclass
class NetworkArchitectureSettings:
    """
    Dataclass for network architecture settings used in the RLlib training process. This class defines the parameters for the neural network architecture used for policy and value function approximation. This includes the hidden layer sizes, activation functions, and whether to use an attention mechanism. These settings help to control the complexity and capacity of the neural network model used in the training process.
    """

    fcnet_hiddens: List[int] = field(
        default_factory=lambda: [512, 512]
    )  #: The hidden layer architecture for the fully connected network. This specifies the number of neurons in each hidden layer of the neural network used for the policy and value function approximation. The default is [512, 512], which means two hidden layers with 512 neurons each. This can be adjusted based on the complexity of the problem and the size of the input state space.
    activation: ActivationFunctionEnum = (
        ActivationFunctionEnum.ReLU
    )  #: The activation function to use for the fully connected network. This specifies the non-linear activation function applied to each neuron in the hidden layers of the neural network. The default is ReLU (Rectified Linear Unit), which is a commonly used activation function in deep learning due to its simplicity and effectiveness. Other options may include Tanh, Sigmoid, etc. This can be adjusted based on the specific requirements of the problem and the architecture of the neural network.
    use_attention: bool = (
        False  #: Whether to use an attention mechanism in the model. This specifies whether to include an attention layer in the neural network architecture. Note, this attends does not attend over the inputs but rather the timestep dimension.
    )
    attention_dim: int = (
        64  #: The dimension of the attention layer. This specifies the size of the output from the attention mechanism if `use_attention` is set to True. The attention dimension determines how many features will be used to represent the output of the attention layer. A larger value may allow for more complex representations but will also increase the computational cost. The default is 64, which is a common choice for many applications.
    )

    @property
    def name(self) -> str:
        return "Network Architecture Settings"

    @classmethod
    def populate_arg_group(cls, args_group):
        args_group.add_argument(
            "--fcnet-hiddens",
            nargs="+",
            type=int,
            default=[512, 512],
            help="Hidden layer architecture for the fully connected network",
        )

        args_group.add_argument(
            "--activation",
            type=ActivationFunctionEnum,
            default=ActivationFunctionEnum.ReLU,
            help="Activation function for the fully connected network",
        )

        args_group.add_argument(
            "--use-attention",
            action="store_true",
            help="Whether to use attention in the model",
        )

        args_group.add_argument(
            "--attention-dim",
            type=int,
            default=64,
            help="The dimension of the attention layer",
        )


@dataclass
class BehaviourCloningSettings:
    """
    Dataclass for behavior cloning settings used in the RLlib training process. This class defines the parameters for behavior cloning. This includes the path to the trajectory data that is collected from the environment and processed into rllib compatible format through conversion script.
    """

    cloning_steps: int = (
        1000000  #: The number of timesteps to clone. This specifies the total number of timesteps to use for behavior cloning. This can be adjusted based on the amount of data available and the desired amount of training data for the behavior cloning process.
    )
    expert_path: Optional[str] = (
        None  #: Path to the original expert data for behavior cloning. This specifies the location of the trajectory data that is collected from the environment. This data needs to be converted into a format compatible with RLlib before it can be used for training the behavior cloning model.
    )
    converted_expert_path: Optional[str] = (
        None  #: Path to the expert data converted to rllib format. This specifies the location of the trajectory data that has been processed into a format compatible with RLlib. This data will be used for training the behavior cloning model.
    )

    @property
    def name(self) -> str:
        return "Behavior Cloning Settings"

    @classmethod
    def populate_arg_group(cls, args_group):
        args_group.add_argument(
            "--expert-path",
            type=str,
            default=None,
            help="Path to the original expert data for behavior cloning from the environment",
        )

        args_group.add_argument(
            "--converted-expert-path",
            type=str,
            default="./ckpt",
            help="Path to store the expert data for behavior cloning",
        )

        args_group.add_argument(
            "--cloning-steps",
            type=int,
            default=1000000,
            help="Number of timesteps to clone.",
        )


@dataclass
class RLlibScriptArgs(ScriptArgs):
    """
    Top level dataclass for RLlib script arguments. This class aggregates all the settings required for configuring the RLlib training process. It includes settings for training, algorithms, logging, resuming from checkpoints, network architecture, and resource allocation. This allows for a comprehensive configuration of the RLlib training job in a structured manner.
    """

    # Training Arguments
    training_settings: TrainingSettings = field(default_factory=TrainingSettings)

    # Training Algorithm Arguments
    algorithm_settings: Union[PPOSettings, APPOSettings, IMPALASettings] = field(
        default_factory=PPOSettings
    )

    # Logging Arguments
    logging_settings: LoggingSettings = field(default_factory=LoggingSettings)

    # Resume Arguments
    resume_settings: ResumeSettings = field(default_factory=ResumeSettings)

    # Network Architecture Arguments
    network_architecture_settings: NetworkArchitectureSettings = field(
        default_factory=NetworkArchitectureSettings
    )

    # Resource Arguments
    resource_settings: ResourceSettings = field(default_factory=ResourceSettings)

    # Behaviour Cloning Arguments
    behaviour_cloning_settings: BehaviourCloningSettings = field(
        default_factory=BehaviourCloningSettings
    )

    # List of plugins
    plugins: List[RLLibLauncherExtension] = field(default_factory=lambda: [])
