# Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
"""
Implementation of a DictionarySpace, a space representing a string keyed dictionary of other spaces.
"""
from collections import OrderedDict
from functools import cached_property
from typing import Dict, List, Union
import gymnasium
import schola.generated.Spaces_pb2 as proto_spaces
import schola.generated.Points_pb2 as proto_points
from .base import UnrealSpace
import numpy as np
import logging
from .discrete import DiscreteSpace, MultiDiscreteSpace
from .binary import MultiBinarySpace
from .box import BoxSpace
from .base import get_space_shape_as_int, merge_space_shape


class DictSpace(gymnasium.spaces.Dict):
    """
    A Space representing a dictionary of spaces.

    Parameters
    ----------
    space_dict : Dict[str, gymnasium.spaces.Space]
        The dictionary of spaces to be represented.

    Attributes
    ----------
    spaces : Dict[str, gymnasium.spaces.Space]
        The dictionary of spaces represented by this object.

    See Also
    --------
    gymnasium.spaces.Dict : The gym space object that this class is analogous to.
    proto_spaces.DictSpace : The protobuf representation of this space.
    """

    def __init__(self, space_dict=None):
        super().__init__(space_dict)
        self._shape = merge_space_shape(self.spaces.values())

    def fill_proto(self, msg: proto_points.DictPoint, action):
        for name, space in self.spaces.items():
            space.fill_proto(msg.values.add(), action[name])

    @cached_property
    def shapes(self):
        """
        Get the shapes of the subspaces in the dictionary space.

        Returns
        -------
        Dict[str, Tuple[int]]
            A dictionary of the shapes of the subspaces in the dictionary space

        Examples
        --------
        >>> space = DictSpace({"a": BoxSpace(0, 1, shape=(2,)), "b": DiscreteSpace(3)})
        >>> space.shapes
        {'a': 2, 'b': 1}
        """
        ret_val = dict()
        for name, space in self.spaces.items():
            ret_val[name] = get_space_shape_as_int(space)
        return ret_val

    @classmethod
    def from_proto(cls, message):
        subspace_dict = OrderedDict()
        for name,value in zip(message.labels,message.values):
            if value.HasField(BoxSpace._name):
                new_entry = BoxSpace.from_proto(value.box_space)
            elif value.HasField(DiscreteSpace._name):
                new_entry = MultiDiscreteSpace.from_proto(value.discrete_space)
            elif value.HasField(MultiBinarySpace._name):
                new_entry = MultiBinarySpace.from_proto(value.binary_space)
            subspace_dict[name] = new_entry
        return DictSpace(subspace_dict)

    def to_normalized(self):
        """
        Normalize this dictionary space by normalizing all of the subspaces in this dictionary space.

        Returns
        -------
        DictSpace
            The normalized dictionary space. A modified version of the space this method is called on

        Examples
        --------
        >>> space = DictSpace({"a": BoxSpace([0,0],[2,2]), "b": DiscreteSpace(3)})
        >>> space.to_normalized()
        Dict('a': Box(0.0, 2.0, (2,), float32), 'b': Discrete(3))
        """
        for key, value in self.spaces.items():
            value.to_normalized()
        return self

    def process_data(self, msg: proto_points.DictPoint):
        return {
            name: space.process_data(point_msg)
            for name, space, point_msg in zip(*zip(*self.spaces.items()), msg.values)
        }

    @property
    def has_only_one_fundamental_type(self):
        """
        Check if all the subspaces in the dictionary space are of the same fundamental type.

        Returns
        -------
        bool
            True if all the subspaces are of the same fundamental type, False otherwise

        Examples
        --------
        >>> space = DictSpace({"a": BoxSpace([0,0],[2,2]), "b": DiscreteSpace(3)})
        >>> space.has_only_one_fundamental_type
        False

        >>> space = DictSpace({"a": BoxSpace([0,0],[2,2]), "b": BoxSpace([0,0],[2,2])})
        >>> space.has_only_one_fundamental_type
        True

        >>> space = DictSpace({"a": DiscreteSpace(3), "b": MultiDiscreteSpace([3,3])})
        >>> space.has_only_one_fundamental_type
        True
        """
        fundamental_type = None
        for key, value in self.spaces.items():
            if fundamental_type is None:
                fundamental_type = type(value)
            elif fundamental_type in [DiscreteSpace, MultiDiscreteSpace]:
                if not isinstance(value, (DiscreteSpace, MultiDiscreteSpace)):
                    return False
            else:
                if not isinstance(value, fundamental_type):
                    return False
        return True

    def simplify(self) -> UnrealSpace:
        """
        Simplify the dictionary space by merging subspaces of the same fundamental type, if possible.

        Returns
        -------
        gymnasium.spaces.Space
            The simplified space

        Examples
        --------
        >>> space = DictSpace({"a": BoxSpace([0,0],[2,2]), "b": BoxSpace([0,0],[2,2])})
        >>> space.simplify()
        Box(0.0, 2.0, (4,), float32)

        >>> space = DictSpace({"a": DiscreteSpace(4), "b": BoxSpace([0,0],[2,2])})
        >>> space.simplify()
        Dict('a': Discrete(4), 'b': Box(0.0, 2.0, (2,), float32))

        >>> space = DictSpace({"a": DiscreteSpace(4)})
        >>> space.simplify()
        Discrete(4)
        """
        # Only one space so simplify to it
        if len(self.spaces) == 1:
            return next(iter(self.spaces.values()))

        # We can merge matching spaces
        elif self.has_only_one_fundamental_type:
            spaces = list(self.spaces.values())
            return type(spaces[0]).merge(*spaces)

        else:
            return self
