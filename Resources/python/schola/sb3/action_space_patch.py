# Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Tools for adapting Stable Baselines 3 PPO implementation to work with dictionary action spaces.
"""
from collections import OrderedDict, defaultdict
from functools import cached_property, reduce
from stable_baselines3.common.distributions import (
    Distribution,
    DiagGaussianDistribution,
    MultiCategoricalDistribution,
    CategoricalDistribution,
    BernoulliDistribution,
    StateDependentNoiseDistribution,
)
from stable_baselines3.common.distributions import make_proba_distribution
from stable_baselines3 import PPO
import stable_baselines3.common.base_class as base
import stable_baselines3.common.buffers as buffers
import stable_baselines3.common.preprocessing as preprocessing
import stable_baselines3.common.policies as policies
import torch as th
import torch.nn as nn
from gymnasium import spaces
from typing import Callable, Dict, Iterable, Tuple, Union, Optional, Type, Any
import warnings

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule


def reshape_nonbatch(tensor: th.Tensor) -> th.Tensor:
    batch_dim = tensor.shape[0]
    return tensor.view(batch_dim, -1)


# this base class is a lie we use to bypass awful design in SB3 (deep seated checks that our distribution is one of 4 distributions)
class HybridDistribution(DiagGaussianDistribution):
    """
    A composite distribution supporting discrete and continuous sub-distributions.

    Parameters
    ----------
    distributions : OrderedDict[str,Distribution]
        A dictionary of distributions to use for the composite distribution.
    discrete_norm_factor : float, default=1.0
        The normalization factor for discrete actions, by default 1.0
    continuous_norm_factor : float, default=1.0
        The normalization factor for continuous actions, by default 1.0

    Attributes
    ----------
    distributions : OrderedDict[str,Distribution]
        A dictionary of distributions to use for the composite distribution.
    """

    # we could make this take an action space and use that to create the various dims we need
    def __init__(
        self,
        distributions: OrderedDict,
        discrete_norm_factor=1.0,
        continuous_norm_factor=1.0,
    ):
        self.distributions: OrderedDict[str, Distribution] = distributions
        self._discrete_norm_factor = discrete_norm_factor
        self._continuous_norm_factor = continuous_norm_factor

    @cached_property
    def action_dims(self) -> Dict[str, int]:
        """
        The size of the action tensor corresponding to each branch of the distribution.

        Returns
        -------
        Dict[str,int]
            A dictionary mapping branch of the distribution to the size of the action tensor corresponding to that branch.
        """
        action_dims = {}
        # We take in OneHot but output discrete value so action is number of values and layer is sum of values (e.g.OneHot)
        for name, dist in self.distributions.items():
            if isinstance(dist, MultiCategoricalDistribution):
                action_dims[name] = len(dist.action_dims)
            elif isinstance(dist, CategoricalDistribution):
                action_dims[name] = 1
            else:
                # not all of the classes use action_dim or action_dims, so try both
                action_dims[name] = (
                    dist.action_dims
                    if hasattr(dist, "action_dims")
                    else dist.action_dim
                )
        return action_dims

    @cached_property
    def action_dim(self) -> int:
        """
        The size of the action tensor corresponding to this distribution.

        Returns
        -------
        int
            The size of the action tensor corresponding to this distribution.
        """
        return sum(self.action_dims.values())

    # Note: We treat everything as having a dimension of size 0 by default
    @cached_property
    def log_std_dims(self) -> Dict[str, int]:
        """
        The number of neurons required for the log standard deviation of each branch.

        Returns
        -------
        Dict[str,int]
            A dictionary mapping branch of the distribution to the number of neurons required for the log standard deviation.
        """
        log_std_dims = defaultdict(lambda: 0)
        for name, dist in self.distributions.items():
            if isinstance(
                dist, (DiagGaussianDistribution, StateDependentNoiseDistribution)
            ):
                log_std_dims[name] = dist.action_dim
        return log_std_dims

    @cached_property
    def log_std_dim(self) -> int:
        """
        The number of neurons required for the log standard deviation.

        Returns
        -------
        int
            The number of neurons required for the log standard deviation.
        """
        return sum(self.log_std_dims.values())

    @cached_property
    def layer_dims(self) -> Dict[str, int]:
        """
        The number of neurons required for each branch of the distribution.

        Returns
        -------
        Dict[str,int]
            A dictionary mapping branch of the distribution to the number of neurons required.
        """
        layer_dims = {}
        for name, dist in self.distributions.items():
            if isinstance(dist, MultiCategoricalDistribution):
                # We take in OneHot but output discrete value so action gets length and layer gets sum
                layer_dims[name] = sum(dist.action_dims)
            else:
                # not all of the classes use action_dim or action_dims, so try both
                layer_dims[name] = (
                    dist.action_dims
                    if hasattr(dist, "action_dims")
                    else dist.action_dim
                )
        return layer_dims

    @cached_property
    def layer_dim(self) -> int:
        """
        The neurons required for this distribution.

        Returns
        -------
        int
            The number of neurons required for this distribution
        """
        return sum(self.layer_dims.values())

    def proba_distribution_net(self, latent_dim, log_std_init: float = 0.0):
        mean_actions = nn.Linear(latent_dim, self.layer_dim)
        log_std = nn.Parameter(
            th.ones(self.log_std_dim) * log_std_init, requires_grad=True
        )
        return mean_actions, log_std

    def proba_distribution(self, mean_actions: th.Tensor, log_std: th.Tensor):
        action_view_start = 0
        log_std_view_start = 0
        for name, dist in self.distributions.items():
            # Note this is a closure to make life easier
            if isinstance(
                dist, (DiagGaussianDistribution, StateDependentNoiseDistribution)
            ):
                dist.proba_distribution(
                    mean_actions[
                        :, action_view_start : action_view_start + self.layer_dims[name]
                    ],
                    log_std[
                        log_std_view_start : log_std_view_start
                        + self.log_std_dims[name]
                    ],
                )
            else:
                dist.proba_distribution(
                    mean_actions[
                        :, action_view_start : action_view_start + self.layer_dims[name]
                    ]
                )
            # see log_std_dims for why this is valid
            action_view_start += self.layer_dims[name]
            log_std_view_start += self.log_std_dims[name]
        return self

    def action_generator(self, action: th.Tensor) -> Iterable[th.Tensor]:
        """
        Takes an Action Sampled from this distribution and generates the actions corresponding to each branch
        of the distribution (e.g. if we have 2 box spaces, it generates a sequence of 2 values sampled from those distributions)

        Parameters
        ----------
        action : th.Tensor
            The action to generate the sub-actions from.

        Yields
        -------
        th.Tensor
            The sub-action corresponding to a branch of the distribution.

        """
        action_view_start = 0
        for name, dist in self.distributions.items():
            curr_action = action[
                :, action_view_start : action_view_start + self.action_dims[name]
            ]
            if isinstance(dist, CategoricalDistribution):
                curr_action = curr_action.view(
                    -1
                )  # Get rid of any non-batch dimensions
            yield curr_action
            action_view_start += self.action_dims[name]
        return

    def log_prob(self, actions) -> th.Tensor:
        return reduce(
            lambda x, y: x + y,
            map(
                lambda x: x[1].log_prob(x[0]),
                zip(self.action_generator(actions), self.distributions.values()),
            ),
        )

    def map_dists(self, func: Callable[[Distribution], Any], normalize: bool = False):
        """
        Maps a function over the distributions in the composite distribution.

        Parameters
        ----------
        func : Callable[[Distribution], Any]
            The function to map over the distributions.
        normalize : bool, optional
            Whether to normalize the output of the function using the norm factors, by default False

        """

        def _inner(key):
            dist = self.distributions[key]
            result = func(dist)
            if normalize:
                if isinstance(
                    dist,
                    (
                        CategoricalDistribution,
                        MultiCategoricalDistribution,
                        BernoulliDistribution,
                    ),
                ):
                    result = result * self._discrete_norm_factor
                elif isinstance(
                    dist, (DiagGaussianDistribution, StateDependentNoiseDistribution)
                ):
                    # handle log_std boys
                    result = result * self._continuous_norm_factor
            return result

        return map(_inner, self.distributions)

    def entropy(self) -> th.Tensor:
        entropy = reduce(
            lambda x, y: x + y, self.map_dists(lambda x: x.entropy(), True)
        )
        return entropy

    # returns [1xN] where N is total size of Flattened Distributions
    def sample(self) -> th.Tensor:
        output = th.cat(
            [
                sample
                for sample in self.map_dists(lambda x: reshape_nonbatch(x.sample()))
            ],
            dim=1,
        )
        return output

    def mode(self):
        mode = th.cat(
            [
                mode
                for mode in self.map_dists(lambda dist: reshape_nonbatch(dist.mode()))
            ],
            dim=1,
        )
        return mode

    # no changes vs DiagGaussianDistribution but kept incase there are later
    def actions_from_params(
        self, action_logits: th.Tensor, log_std: th.Tensor, deterministic: bool = False
    ) -> th.Tensor:
        self.proba_distribution(action_logits, log_std)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(
        self, mean_actions: th.Tensor, log_std: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(mean_actions, log_std)
        return actions, self.log_prob(actions)


cached_make_proba_dist = make_proba_distribution


def make_hybrid_dist(
    action_space: spaces.Dict,
    use_sde: bool = False,
    discrete_norm_factor=1.0,
    continuous_norm_factor=1.0,
):
    """
    Create a hybrid distribution from a dictionary of action spaces.

    Parameters
    ----------
    action_space : spaces.Dict
        The dictionary of action spaces to create the distribution from.
    use_sde : bool, optional
        Whether to use state dependent noise, by default False
    discrete_norm_factor : float, optional
        The normalization factor for discrete actions, by default 1.0
    continuous_norm_factor : float, optional
        The normalization factor for continuous actions, by default 1.0
    """
    distributions = OrderedDict(
        [
            (key, cached_make_proba_dist(action_space[key], use_sde))
            for key in action_space
        ]
    )
    return HybridDistribution(
        distributions, discrete_norm_factor, continuous_norm_factor
    )


def patched_with_norm(discrete_norm_factor=1.0, continuous_norm_factor=1.0):
    def patched_make_proba_dist(action_space, use_sde=False, dist_kwargs=None):
        if isinstance(action_space, spaces.Dict):
            return make_hybrid_dist(
                action_space, use_sde, discrete_norm_factor, continuous_norm_factor
            )
        else:
            return cached_make_proba_dist(action_space, use_sde, dist_kwargs)

    return patched_make_proba_dist


# Below code is adapted from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/callbacks.py

# The MIT License
#
# Copyright (c) 2019 Antonin Raffin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Modifications Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.


class PatchedPPO(PPO):

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        OnPolicyAlgorithm.__init__(
            self,
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
                spaces.Dict,
            ),
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        if _init_setup_model:
            PPO._setup_model(self)


# End of Adapted Code


cached_get_action_dim = preprocessing.get_action_dim


def patched_get_action_dim(action_space):
    if isinstance(action_space, spaces.Dict):
        print("using hybrid action dist")
        return sum([cached_get_action_dim(action_space[key]) for key in action_space])
    else:
        return cached_get_action_dim(action_space)


class ActionSpacePatch:
    """
    A context manager that patches the stable baselines3 library to support custom action spaces.
    This is done by overriding the make_proba_distribution function in the stable baselines3 library
    with a custom function that supports custom action spaces. Currently only works with PPO.
    discrete and continuous actions are balanced with the `discrete_norm_factor` and `continuous_norm_factor` respectively.

    Parameters
    ----------
    globs : Dict
        The globals dictionary of the calling module.
    discrete_norm_factor : float, optional
        The normalization factor for discrete actions, by default 1.0
    continuous_norm_factor : float, optional
        The normalization factor for continuous actions, by default 1.0

    Attributes
    ----------
    globs : Dict
        The globals dictionary of the calling module.
    _continuous_norm_factor : float
        The normalization factor for continuous actions.
    _discrete_norm_factor : float
        The normalization factor for discrete actions.
    """

    def __init__(self, globs, discrete_norm_factor=1.0, continuous_norm_factor=1.0):
        self.globs = globs
        self._continuous_norm_factor = continuous_norm_factor
        self._discrete_norm_factor = discrete_norm_factor

    def __enter__(self):
        """
        Patch the stable baselines3 library to support custom action spaces.
        """

        policies.__dict__["make_proba_distribution"] = patched_with_norm(
            self._discrete_norm_factor, self._continuous_norm_factor
        )

        self.globs["PPO"] = PatchedPPO

        buffers.__dict__["get_action_dim"] = patched_get_action_dim

    def __exit__(self, type, value, traceback):
        """
        Unpatch the stable baselines3 library.

        Parameters
        ----------
        type : type
            The type of the exception if one was raised.
        value : Exception
            The exception that was raised, if one was raised.
        traceback : Traceback
            The traceback of the exception, if one was raised.
        """
        base.__dict__["make_proba_distribution"] = cached_make_proba_dist
        preprocessing.__dict__["get_action_dim"] = cached_get_action_dim
        self.globs["PPO"] = PPO
