# Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Base classes for Schola spaces.
"""

from typing import Any, Dict, List, Tuple, Type
import gymnasium
from gymnasium.spaces import Space
import schola.generated.Spaces_pb2 as proto_spaces
import schola.generated.Points_pb2 as proto_points
import numpy as np
import logging


class UnrealSpace:
    """
    A base class for all spaces in Schola, providing a common interface for converting between protobuf messages and pythonic representations.

    Attributes
    ----------
    _name : str
        The name of the space.

    See Also
    --------
    gymnasium.spaces.Space : The gym space object that this class is analogous to.
    """

    proto_space: Type[proto_spaces.FundamentalSpace] = (
        None  #: A class variable containing the protobuf representation of the space.
    )

    @classmethod
    def from_proto(cls, message) -> "UnrealSpace":
        """
        Create a Space Object from a protobuf representation.

        Parameters
        ----------
        message : proto_space
            The protobuf message to convert.

        Returns
        -------
        UnrealSpace
            The Space subclass created from the protobuf message
        """
        ...

    @classmethod
    def is_empty_definition(cls, message) -> bool:
        """
        Returns True iff this space has magnitude 0.

        Parameters
        ----------
        message : proto_space
            The protobuf message to check for emptiness.

        Returns
        -------
        bool
            True iff the space is empty.
        """
        ...

    def process_data(self, msg: proto_points.FundamentalPoint) -> np.ndarray:
        """
        Convert a protobuf message corresponding to a point in this space to a pythonic representation.

        Parameters
        ----------
        msg : proto_points.FundamentalPoint
            The protobuf message to convert.

        Returns
        -------
        np.ndarray
            The pythonic representation of the point.
        """
        ...

    def fill_proto(self, msg: proto_points.FundamentalPoint, value: Any) -> None:
        """
        Convert a python representation of point in this space to a protobuf message. Mutates msg with the result.

        Parameters
        ----------
        msg : proto_points.FundamentalPoint
            The protobuf message to fill.
        value : Any
            The pythonic representation of the point.
        """
        ...

    def to_normalized(self) -> "UnrealSpace":
        """
        Returns a normalized version of the space. Is a noop if a space subclass does not implement `to_normalized`.

        Returns
        -------
        UnrealSpace
            The normalized space.
        """
        return self

    def __len__(self) -> int:
        """
        Returns the length of the space.
        """
        ...

    @classmethod
    def merge(cls, *spaces: List["UnrealSpace"]) -> "UnrealSpace": ...


def get_space_shape_as_int(space: Space) -> int:
    """
    Get the shape of a space as an integer.

    Parameters
    ----------
    space : Space
        The space to get the shape of.

    Returns
    -------
    int
        The shape of the space as an integer.
    """
    # handle discrete spaces which have shape = (,)
    if len(space.shape) == 0:
        return 1
    else:
        return sum(space.shape)


def merge_space_shape(spaces: List[Space]) -> Tuple[int]:
    """
    Merge the shapes of multiple spaces into a single shape.

    Parameters
    ----------
    spaces : List[Space]
        The spaces to merge.

    Returns
    -------
    Tuple[int]
        The merged shape.
    """
    shape_dim = 0
    for space in spaces:
        # handle discrete spaces which have shape = (,)
        shape_dim += get_space_shape_as_int(space)
    return (shape_dim,)
