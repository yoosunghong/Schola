# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

from typing import List, Type, Union, Any
from schola.sb3.action_space_patch import PatchedPPO
from stable_baselines3 import PPO, SAC
from schola.scripts.common import (
    ActivationFunctionEnum,
    ScriptArgs,
    Sb3LauncherExtension,
)
from dataclasses import dataclass, field


@dataclass
class PPOSettings:
    """
    Dataclass for configuring the settings of the Proximal Policy Optimization (PPO) algorithm. This includes parameters for the learning process, such as learning rate, batch size, number of steps, and other hyperparameters that control the behavior of the PPO algorithm.
    """

    learning_rate: float = 0.0003  #: Learning rate for the optimizer.
    n_steps: int = (
        2048  #: Number of steps to run for each environment per update. This is the number of timesteps collected before updating the policy.
    )
    batch_size: int = (
        64  #: Minibatch size for each update. This is the number of timesteps used in each batch for training the policy. Must be a divisor of `n_steps`.
    )
    n_epochs: int = (
        10  #: Number of epochs to update the policy. This is the number of times the model will iterate over the collected data during training. More epochs can lead to better convergence but also overfitting.
    )
    gamma: float = (
        0.99  #: Discount factor for future rewards. This determines how much the agent values future rewards compared to immediate rewards. A value of 0.99 means that future rewards are discounted by 1% per time step.
    )
    gae_lambda: float = (
        0.95  #: Lambda parameter for Generalized Advantage Estimation (GAE). This parameter helps to balance bias and variance in the advantage estimation. A value of 1.0 corresponds to standard advantage estimation, while lower values will reduce variance but may introduce bias.
    )
    clip_range: float = (
        0.2  #: Clipping range for the policy update. This is the maximum amount by which the new policy can differ from the old policy during training. This helps to prevent large updates that can destabilize training.
    )
    normalize_advantage: bool = (
        True  #: Whether to normalize the advantages. Normalizing the advantages can help to stabilize training by ensuring that they have a mean of 0 and a standard deviation of 1. This can lead to more consistent updates to the policy.
    )
    ent_coef: float = (
        0.0  #: Coefficient for the entropy term in the loss function. This encourages exploration by adding a penalty for certainty in the policy's action distribution. A higher value will encourage more exploration, while a lower value will make the policy more deterministic. Set to 0.0 to disable entropy regularization.
    )
    vf_coef: float = (
        0.5  #: Coefficient for the value function loss in the overall loss function. This determines how much weight is given to the value function loss compared to the policy loss. A higher value will put more emphasis on accurately estimating the value function, while a lower value will prioritize the policy update.
    )
    max_grad_norm: float = (
        0.5  #: Maximum gradient norm for clipping. This is used to prevent exploding gradients by scaling down the gradients if their norm exceeds this value. This can help to stabilize training, especially in environments with high variance in the rewards or gradients.
    )
    use_sde: bool = (
        False  #: Whether to use State Dependent Exploration (SDE). This can help to improve exploration by adapting the exploration noise based on the current state of the environment. When set to True, it will use SDE for exploration instead of the standard exploration strategy.
    )
    sde_sample_freq: int = (
        -1
    )  #: Frequency at which to sample the SDE noise. This determines how often the noise is sampled when using State Dependent Exploration (SDE). A value of -1 means that it will sample the noise at every step, while a positive integer will specify the number of steps between samples. This can help to control the exploration behavior of the agent.

    @property
    def constructor(self) -> Type[PPO]:
        return PatchedPPO

    @property
    def name(self) -> str:
        return "PPO"

    @property
    def critic_type(self) -> str:
        return "vf"


@dataclass
class SACSettings:
    """
    Dataclass for configuring the settings of the Soft Actor-Critic (SAC) algorithm. This includes parameters for the learning process, such as learning rate, buffer size, batch size, and other hyperparameters that control the behavior of the SAC algorithm.
    """

    learning_rate: float = (
        0.0003  #: Learning rate for the optimizer. This controls how much to adjust the model parameters in response to the estimated error each time the model weights are updated. A lower value means slower learning, while a higher value means faster learning.
    )
    buffer_size: int = (
        1000000  #: Size of the replay buffer. This is the number of transitions (state, action, reward, next state) that can be stored in the buffer. A larger buffer allows for more diverse samples to be used for training, which can improve performance but also increases memory usage.
    )
    learning_starts: int = (
        100  #: Number of timesteps before learning starts. This is the number of steps to collect in the replay buffer before the first update to the policy. This allows the agent to gather initial experience and helps to stabilize training by ensuring that there are enough samples to learn from.
    )
    batch_size: int = (
        256  #: Minibatch size for each update. This is the number of samples drawn from the replay buffer to perform a single update to the policy. A larger batch size can lead to more stable updates but requires more memory. Must be less than or equal to `buffer_size`.
    )
    tau: float = (
        0.005  #: Soft update parameter for the target networks. This controls how much the target networks are updated towards the main networks during training. A smaller value (e.g., 0.005) means that the target networks are updated slowly, which can help to stabilize training. This is typically a small value between 0 and 1.
    )
    gamma: float = (
        0.99  #: Discount factor for future rewards. This determines how much the agent values future rewards compared to immediate rewards. A value of 0.99 means that future rewards are discounted by 1% per time step. This is important for balancing the trade-off between short-term and long-term rewards in reinforcement learning.
    )
    train_freq: int = (
        1  #: Frequency of training the policy. This determines how often the model is updated during training. A value of 1 means that the model is updated every time step, while a higher value (e.g., 2) means that the model is updated every other time step. This can help to control the trade-off between exploration and exploitation during training.
    )
    gradient_steps: int = (
        1  #: Number of gradient steps to take during each training update. This specifies how many times to update the model parameters using the sampled minibatch from the replay buffer. A value of 1 means that the model is updated once per training step, while a higher value (e.g., 2) means that the model is updated multiple times. This can help to improve convergence but may also lead to overfitting if set too high.
    )
    action_noise: Any = (
        None  #: Action noise to use for exploration. This can be a callable function or a noise process (e.g., Ornstein-Uhlenbeck) that adds noise to the actions taken by the policy to encourage exploration. This is important in continuous action spaces to help the agent explore different actions and avoid getting stuck in local optima. If set to None, no noise will be added to the actions.
    )
    replay_buffer_class: Any = (
        None  #: Class to use for the replay buffer. This allows for customization of the replay buffer used for training. By default, it will use the standard `ReplayBuffer` class provided by Stable Baselines3. However, you can specify a custom class that inherits from `ReplayBuffer` to implement your own functionality or behavior for storing and sampling transitions.
    )
    replay_buffer_kwargs: dict = (
        None  #: Additional keyword arguments to pass to the replay buffer constructor. This allows for further customization of the replay buffer's behavior and settings when it is instantiated. For example, you can specify parameters like `buffer_size`, `seed`, or any other parameters supported by your custom replay buffer class. This can help to tailor the replay buffer to your specific needs or environment requirements.
    )
    optimize_memory_usage: bool = (
        False  #: Whether to optimize memory usage for the replay buffer. When set to True, it will use a more memory-efficient implementation of the replay buffer, which can help to reduce memory consumption during training. This is particularly useful when working with large environments or limited hardware resources. Note that this may slightly affect the performance of the training process, as it may introduce some overhead in accessing the samples.
    )
    ent_coef: Any = (
        "auto"  #: Coefficient for the entropy term in the loss function. This encourages exploration by adding a penalty for certainty in the policy's action distribution. A higher value will encourage more exploration, while a lower value will make the policy more deterministic. When set to 'auto', it will automatically adjust the coefficient based on the average entropy of the actions taken by the policy. This can help to balance exploration and exploitation during training.
    )
    target_update_interval: int = (
        1  #: Interval for updating the target networks. This determines how often the target networks are updated with the main networks' weights. A value of 1 means that the target networks are updated every training step, while a higher value (e.g., 2) means that they are updated every other step. This can help to control the stability of training by ensuring that the target networks are kept up-to-date with the latest policy parameters.
    )
    target_entropy: Any = (
        "auto"  #: Target entropy for the entropy regularization. This is used to encourage exploration by setting a target for the average entropy of the actions taken by the policy. When set to 'auto', it will automatically calculate the target entropy based on the dimensionality of the action space (e.g., -dimensionality of the action space). This helps to balance exploration and exploitation during training by encouraging the agent to explore more diverse actions.
    )
    use_sde: bool = (
        False  #: Whether to use State Dependent Exploration (SDE). This can help to improve exploration by adapting the exploration noise based on the current state of the environment. When set to True, it will use SDE for exploration instead of the standard exploration strategy. This can lead to more efficient exploration in complex environments, but may also introduce additional computational overhead.
    )
    sde_sample_freq: int = (
        -1
    )  #: Frequency at which to sample the SDE noise. This determines how often the noise is sampled when using State Dependent Exploration (SDE). A value of -1 means that it will sample the noise at every step, while a positive integer will specify the number of steps between samples. This can help to control the exploration behavior of the agent. A higher frequency can lead to more diverse exploration, while a lower frequency may lead to more stable but less exploratory behavior.

    @property
    def constructor(self) -> Type[SAC]:
        return SAC

    @property
    def name(self) -> str:
        return "SAC"

    @property
    def critic_type(self) -> str:
        return "qf"


@dataclass
class SB3ScriptArgs(ScriptArgs):
    """
    Top level dataclass for configuring the script arguments used in the SB3 launcher. This dataclass extends `ScriptArgs` and includes various settings for training algorithms, logging, and other configurations. It allows for easy customization of the training process by specifying parameters such as timesteps, logging options, network architectures, and algorithm-specific settings.
    """

    # General Arguments
    timesteps: int = (
        3000  #: Total number of timesteps to train the agent. This is the total number of environment steps that will be used for training. This should be set based on the complexity of the environment and the desired training duration. A higher value will typically lead to better performance but will also increase training time.
    )

    #   Misc Arguments
    pbar: bool = (
        False  #: Whether to display a progress bar during training. Requires TQDM and Rich to be installed.
    )
    disable_eval: bool = (
        False  #: Whether to disable running evaluation after training. When set to True, it will skip evaluation after training completes.
    )

    # Logging Arguments
    enable_tensorboard: bool = False  #: Whether to enable TensorBoard logging.
    log_dir: str = "./logs"  #: Directory to save TensorBoard logs.
    log_freq: int = (
        10  #: Frequency of logging training metrics to TensorBoard. This determines how often (in terms of training steps) the training metrics will be logged to TensorBoard. A value of 10 means that every 10 training steps, the metrics will be recorded.
    )
    callback_verbosity: int = (
        0  #: Verbosity level for callbacks. This controls the level of detail in the output from any callbacks used during training.
    )
    schola_verbosity: int = (
        0  #: Verbosity level for Schola-specific logging. This controls the level of detail in the output from Schola-related components during training.
    )
    sb3_verbosity: int = (
        1  #: Verbosity level for Stable Baselines3 logging. This controls the level of detail in the output from Stable Baselines3 components during training.
    )

    # Checkpoint Arguments
    save_replay_buffer: bool = (
        False  #: Whether to save the replay buffer when saving a checkpoint. This allows for resuming training from the same state of the replay buffer.
    )
    save_vecnormalize: bool = (
        False  #: Whether to save the vector normalization statistics when saving a checkpoint. This is useful for environments where observations need to be normalized, and it allows for consistent normalization when resuming training.
    )

    # Resume Arguments
    resume_from: str = (
        None  #: Path to a saved model to resume training from. This allows for continuing training from a previously saved checkpoint. The path should point to a valid model file created by Stable Baselines3. If set to None, training will start from scratch.
    )
    load_vecnormalize: str = (
        None  #: Path to a saved vector normalization statistics file to load when resuming training. This allows for loading the normalization statistics from a previous training session, ensuring that the observations are normalized consistently when resuming training. If set to None, it will not load any vector normalization statistics.
    )
    load_replay_buffer: str = (
        None  #: Path to a saved replay buffer to load when resuming training. This allows for loading a previously saved replay buffer, which can be useful for continuing training with the same set of experiences. The path should point to a valid replay buffer file created by Stable Baselines3. If set to None, it will not load any replay buffer, and a new one will be created instead.
    )
    reset_timestep: bool = (
        False  #: Whether to reset the internal timestep counter when resuming training from a saved model. When set to True, it will reset the timestep counter to 0.
    )

    # Network Architecture Arguments
    policy_parameters: List[int] = (
        None  #: A list of layer widths representing the policy network architecture. This defines the number of neurons in each hidden layer of the policy network. For example, [64, 64] would create a policy network with two hidden layers, each containing 64 neurons. If set to None, it will use the default architecture defined by the algorithm.
    )
    critic_parameters: List[int] = (
        None  #: A list of layer widths representing the critic (value function) network architecture. This defines the number of neurons in each hidden layer of the critic network. For example, [64, 64] would create a critic network with two hidden layers, each containing 64 neurons. This is only applicable for algorithms that use a critic (e.g., SAC). If set to None, it will use the default architecture defined by the algorithm.
    )
    activation: ActivationFunctionEnum = (
        ActivationFunctionEnum.ReLU
    )  #: Activation function to use in the policy and critic networks. This determines the non-linear activation function applied to each layer of the neural networks. Common options include ReLU, Tanh, and Sigmoid. The choice of activation function can affect the performance of the model and may depend on the specific characteristics of the environment. Default is ReLU, but you can choose others based on your needs.

    # Behaviour Cloning Arguments
    expert_path: str = (
        None  #: Path to the expert data for behavior cloning. This is the file path to the dataset containing expert demonstrations that will be used for training the agent.
    )
    cloning_epochs: int = (
        10  #: Number of epochs for behavior cloning. This determines how many times the model will iterate over the expert data during training. More epochs can lead to better convergence but also overfitting. The optimal number of epochs may depend on the quality and quantity of the expert data.
    )
    # Training Algorithm Arguments
    batch_size: int = (
        64  #: Minibatch size for training. This is the number of samples drawn from the replay buffer or collected data to perform a single update to the policy. A larger batch size can lead to more stable updates but requires more memory. Must be less than or equal to `buffer_size`.
    )
    minibatch_size: int = 64
    #: Minibatch size for training. This is the number of samples drawn from the replay buffer or collected data to perform a single update to the policy. A larger batch size can lead to more stable updates but requires more memory. Must be less than or equal to `buffer_size`.
    learning_rate: float = (
        0.0003  #: Learning rate for the optimizer. This controls how much to adjust the model parameters in response to the estimated error each time the model weights are updated. A lower value means slower learning, while a higher value means faster learning.
    )
    algorithm_settings: Union[PPOSettings, SACSettings] = field(
        default_factory=PPOSettings
    )  #: The settings for the training algorithm to use. This can be either `PPOSettings` or `SACSettings`, depending on the chosen algorithm. This property allows for easy switching between different algorithms (e.g., PPO or SAC) by simply changing the instance of the settings class. The default is `PPOSettings`, which is suitable for most environments unless specified otherwise.

    # List of plugins
    plugins: List[Sb3LauncherExtension] = field(
        default_factory=lambda: []
    )  #: A list of Plugins that can be used to extend the behaviour of launch.py

    @property
    def name_prefix(self):
        return (
            self.name_prefix_override
            if self.name_prefix_override is not None
            else self.algorithm_settings.name.lower()
        )
