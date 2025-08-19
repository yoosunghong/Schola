# Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Implementation of a MultiBinarySpace, a space representing a vector of binary values.
"""
from collections import OrderedDict
from functools import cached_property
from typing import Dict, List, Union
import gymnasium
import schola.generated.Spaces_pb2 as proto_spaces
import schola.generated.Points_pb2 as proto_points
import numpy as np
import logging
from .base import UnrealSpace


class MultiBinarySpace(gymnasium.spaces.MultiBinary, UnrealSpace):
    """
    A Space representing a vector of binary values.

    Parameters
    ----------
    n : int
        The number of binary values in the space.

    Attributes
    ----------
    shape : Tuple[int]
        The shape of the space.
    n : int
        The number of binary values in the space.

    See Also
    --------
    gymnasium.spaces.MultiBinary : The gym space object that this class is analogous to.
    proto_spaces.BinarySpace : The protobuf representation of this space.
    """

    proto_space = proto_spaces.BinarySpace
    _name = "binary_space"

    def __init__(self, n: int):
        super().__init__(n=n)

    def to_normalized(self):
        """
        Cannot normalize a binary space, so return self.
        """
        return self

    @classmethod
    def from_proto(cls, message: proto_spaces.BinarySpace):
        return MultiBinarySpace(message.shape)

    @classmethod
    def is_empty_definition(cls, message: proto_spaces.BinarySpace):
        return message.shape == 0

    def fill_proto(self, msg: proto_points.FundamentalPoint, values):
        msg.binary_point.values.extend(values)

    @classmethod
    def merge(cls, *spaces: List["MultiBinarySpace"]) -> "MultiBinarySpace":
        """
        Merge multiple MultiBinarySpaces into a single space.

        Parameters
        ----------
        *spaces : List[MultiBinarySpace]
            The spaces to merge.

        Returns
        -------
        MultiBinarySpace
            The merged space.

        Raises
        ------
        TypeError
            If any of the spaces are not MultiBinarySpaces.

        Examples
        --------
        >>> merged_space = MultiBinarySpace.merge(MultiBinarySpace(3), MultiBinarySpace(4))
        >>> merged_space.n
        7
        """
        try:
            return MultiBinarySpace(sum((space.n for space in spaces)))
        except:
            raise TypeError(
                "can only merge MultiBinarySpaces with other MultiBinarySpaces"
            )

    def process_data(self, msg: proto_points.FundamentalPoint) -> np.ndarray:
        return np.asarray(msg.binary_point.values)

    def __len__(self) -> int:
        return self.shape[0]
