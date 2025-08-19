# Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Implementation of a BoxSpace, a space representing a bounded vector of continuous values.
"""
from typing import Dict, List, Optional, Tuple, Union
import gymnasium
import schola.generated.Spaces_pb2 as proto_spaces
import schola.generated.Points_pb2 as proto_points
import numpy as np
from .base import UnrealSpace


class BoxSpace(gymnasium.spaces.Box, UnrealSpace):
    """
    A Space representing a box in n-dimensional space.

    Parameters
    ----------
    low : Union[float, np.ndarray, List[float]]
        The lower bounds of the box.
    high : Union[float, np.ndarray, List[float]]
        The upper bounds of the box.
    shape : Tuple[int], optional
        The shape of the space.

    Attributes
    ----------
    shape : Tuple[int]
        The shape of the space.

    Note
    ----
    Unlike, the gymnasium Box space, this class does not have a dtype attribute. The dtype is always np.float32.

    See Also
    --------
    gymnasium.spaces.Box : The gym space object that this class is analogous to.
    proto_spaces.BoxSpace : The protobuf representation of this space.
    """

    proto_space = proto_spaces.BoxSpace
    _name = "box_space"

    def __init__(
        self,
        low: Union[float, np.ndarray, List[float]],
        high: Union[float, np.ndarray, List[float]],
        shape: Optional[Tuple[int]] = None,
    ):
        if isinstance(low, list):
            low = np.asarray(low, dtype=np.float32)
        if isinstance(high, list):
            high = np.asarray(high, dtype=np.float32)
        super().__init__(low=low, high=high, shape=shape)

    @classmethod
    def from_proto(cls, message: proto_spaces.BoxSpace) -> "BoxSpace":
        low = []
        high = []
        for dimension in message.dimensions:
            low.append(dimension.low)
            high.append(dimension.high)
        if len(message.shape_dimensions) == 0:
            shape = [len(low)]
        else:
            shape = tuple(message.shape_dimensions)
        low = np.asarray(low, dtype=np.float32).reshape(shape)
        high = np.asarray(high, dtype=np.float32).reshape(shape)
        return BoxSpace(low=low, high=high, shape=shape)

    @classmethod
    def is_empty_definition(cls, message: proto_spaces.BoxSpace) -> bool:
        return len(list(message.dimensions)) == 0

    def fill_proto(self, msg: proto_points.FundamentalPoint, values):
        msg.box_point.values.extend(values)

    def to_normalized(self):
        """
        Normalize the bounds of the space to be between 0 and 1

        Returns
        -------
        BoxSpace
            The normalized space. A modified version of the space this method is called on

        Examples
        --------
        >>> space = BoxSpace([0, 0],[2, 2])
        >>> space.to_normalized() == BoxSpace([0., 0.], [1., 1.])
        True
        """
        self.low = np.zeros_like(self.low)
        self.high = np.ones_like(self.high)
        return self

    @classmethod
    def merge(cls, *spaces: List["BoxSpace"]) -> "BoxSpace":
        """
        Merge multiple BoxSpaces into a single space.

        Parameters
        ----------
        *spaces : List[BoxSpace]
            The spaces to merge.

        Returns
        -------
        BoxSpace
            The merged space.

        Raises
        ------
        TypeError
            If any of the spaces are not BoxSpaces.

        Examples
        --------
        >>> merged_space = BoxSpace.merge(BoxSpace([0,0],[1,1]), BoxSpace([2,2],[3,3]))
        >>> merged_space == BoxSpace([0, 0, 2, 2], [1, 1, 3, 3])
        True
        """
        for space in spaces:
            if not isinstance(space, gymnasium.spaces.Box):
                raise TypeError(f"Cannot merge BoxSpace with {type(space)}")
        low = np.concatenate([space.low for space in spaces])
        high = np.concatenate([space.high for space in spaces])
        # Try and merge on the first axis
        return BoxSpace(low, high)

    def __len__(self) -> int:
        """
        Get the number of dimensions of the space

        Returns
        -------
        int
            The number of dimensions of the space

        Examples
        --------
        >>> space = BoxSpace([0,0],[1,1])
        >>> len(space)
        2
        """
        return self.low.size

    def process_data(self, msg: proto_points.FundamentalPoint) -> np.ndarray:
        output = np.asarray(msg.box_point.values, dtype=np.float32).reshape(self.shape)
        return output
