# Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
"""
Implementation of Discrete and MultiDiscrete spaces, which represent a single and and a vector of bounded discrete values.
"""

import schola.generated.Spaces_pb2 as proto_spaces
import schola.generated.Points_pb2 as proto_points
from schola.core.spaces.base import UnrealSpace
import numpy as np
from typing import List, Union
import gymnasium


def merge_discrete_like_spaces(
    *spaces: List[Union[gymnasium.spaces.Discrete, gymnasium.spaces.MultiDiscrete]]
):
    """
    Merge multiple Discrete or MultiDiscrete spaces into a single MultiDiscrete space.

    Parameters
    ----------
    *spaces : List[Union[gymnasium.spaces.Discrete, gymnasium.spaces.MultiDiscrete]]
        The spaces to merge.

    Returns
    -------
    MultiDiscreteSpace
        The merged space.

    Raises
    ------
    TypeError
        If any of the spaces are not Discrete or MultiDiscrete.

    Examples
    --------
    >>> merged_space = merge_discrete_like_spaces(MultiDiscreteSpace([3,2]), DiscreteSpace(2))
    >>> merged_space == MultiDiscreteSpace([3, 2, 2])
    True

    >>> merged_space = merge_discrete_like_spaces(DiscreteSpace(3), DiscreteSpace(2))
    >>> merged_space == MultiDiscreteSpace([3, 2])
    True

    >>> merged_space = merge_discrete_like_spaces(DiscreteSpace(3))
    >>> merged_space == DiscreteSpace(3)
    True
    """
    nvecs = []
    for space in spaces:
        if isinstance(space, gymnasium.spaces.Discrete):
            nvecs.append([space.n])
        elif isinstance(space, gymnasium.spaces.MultiDiscrete):
            nvecs.append(space.nvec)
        else:
            raise TypeError(f"Can't merge Discrete or MultiDiscrete Space with {space}")
    nvecs = np.concatenate(nvecs)
    return MultiDiscreteSpace(nvecs) if len(nvecs) > 1 else DiscreteSpace(nvecs[0])


class DiscreteSpace(gymnasium.spaces.Discrete, UnrealSpace):
    """
    A Space representing a single discrete value.

    Parameters
    ----------
    n : int
        The number of discrete values in the space. e.g. space is one value in interval [0,n]

    Attributes
    ----------
    n : int
        The number of discrete values in the space.

    See Also
    --------
    gymnasium.spaces.Discrete : The gym space object that this class is analogous to.
    proto_spaces.DiscreteSpace : The protobuf representation of this space.
    """

    proto_space = proto_spaces.DiscreteSpace
    _name = "discrete_space"

    def __init__(self, n):
        super().__init__(n=n)

    @classmethod
    def is_empty_definition(cls, message: proto_spaces.DiscreteSpace):
        return message.size == 0

    def fill_proto(
        self, msg: proto_points.FundamentalPoint, value: Union[int, np.ndarray]
    ):
        if not isinstance(value, int):
            value = int(value.item())
        msg.discrete_point.values.append(value)

    @classmethod
    def merge(
        cls, *spaces: Union["DiscreteSpace", "MultiDiscreteSpace"]
    ) -> "MultiDiscreteSpace":
        """
        Merge multiple DiscreteSpaces into a single space.

        Parameters
        ----------
        *spaces : List[Union[DiscreteSpace, MultiDiscreteSpace]]
            The spaces to merge.

        Returns
        -------
        MultiDiscreteSpace
            The merged space.

        Raises
        ------
        TypeError
            If any of the spaces are not Discrete or MultiDiscrete.

        See Also
        --------
        merge_discrete_like_spaces : Merge multiple Discrete or MultiDiscrete spaces into a single MultiDiscrete space.
        """
        return merge_discrete_like_spaces(*spaces)

    def __len__(self):
        return 1

    def process_data(self, msg: proto_points.FundamentalPoint):
        return next(iter(msg.discrete_point.values))

    def to_normalized(self):
        return self

    def __eq__(self, other):
        return bool(super().__eq__(other))


class MultiDiscreteSpace(gymnasium.spaces.MultiDiscrete, UnrealSpace):
    """
    A Space representing a vector of discrete values.

    Parameters
    ----------
    nvec : List[int]
        The number of discrete values in each dimension of the space.

    Attributes
    ----------
    nvec : List[int]
        The number of discrete values in each dimension of the space.

    See Also
    --------
    gymnasium.spaces.MultiDiscrete : The gym space object that this class is analogous to.
    proto_spaces.MultiDiscreteSpace : The protobuf representation of this space.
    """

    proto_space = proto_spaces.DiscreteSpace
    _name = "discrete_space"

    def __init__(self, nvec: List[int]):
        super().__init__(nvec=nvec)

    @classmethod
    def from_proto(cls, message: proto_spaces.DiscreteSpace) -> "MultiDiscreteSpace":
        high = list(message.high)
        # flatten to discrete
        if len(high) == 1:
            return DiscreteSpace(high[0])
        else:
            return MultiDiscreteSpace(message.high)

    @classmethod
    def merge(
        cls, *spaces: List[Union["MultiDiscreteSpace", "DiscreteSpace"]]
    ) -> "MultiDiscreteSpace":
        """
        Merge multiple DiscreteSpaces into a single space.

        Parameters
        ----------
        *spaces : List[Union[DiscreteSpace, MultiDiscreteSpace]]
            The spaces to merge.

        Returns
        -------
        MultiDiscreteSpace
            The merged space.

        Raises
        ------
        TypeError
            If any of the spaces are not Discrete or MultiDiscrete.

        See Also
        --------
        merge_discrete_like_spaces : Merge multiple Discrete or MultiDiscrete spaces into a single MultiDiscrete space.
        """
        return merge_discrete_like_spaces(*spaces)

    @classmethod
    def is_empty_definition(cls, message: proto_spaces.DiscreteSpace) -> bool:
        high = list(message.high)
        return len(message.high) == 0

    def fill_proto(
        self, msg: proto_points.FundamentalPoint, values: np.ndarray
    ) -> None:
        msg.discrete_point.values.extend([int(value) for value in values])

    def process_data(self, msg: proto_points.FundamentalPoint) -> np.ndarray:
        return np.asarray(msg.discrete_point.values)

    def __len__(self) -> int:
        """
        Get the number of discrete values in the space.

        Returns
        -------
        int
            The number of discrete values in the space.

        Examples
        --------
        >>> space = MultiDiscreteSpace([3, 2])
        >>> len(space)
        2

        >>> space = MultiDiscreteSpace([3])
        >>> len(space)
        1

        >>> space = MultiDiscreteSpace([])
        >>> len(space)
        0
        """
        # edge case where you get shape (0,)
        if self.shape == (0,):
            return 0
        else:
            return sum(self.shape)

    def to_normalized(self) -> "MultiDiscreteSpace":
        return self
