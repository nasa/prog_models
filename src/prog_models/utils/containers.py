# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from ..exceptions import ProgModelTypeError

import numpy as np


class DictLikeMatrixWrapper():
    """
    A container that behaves like a dictionary, but is backed by a numpy array, which is itself directly accessable. This is used for model states, inputs, and outputs- and enables efficient matrix operations.
    
    Arguments
    ---------
    keys: list
        The keys of the dictionary. e.g., model.states or model.inputs
    data: dict or numpy array
        The contained data (e.g., input, state, output). If numpy array should be column vector in same order as keys
    """
    def __init__(self, keys, data):
        self._keys = keys
        if isinstance(data, dict):
            self.matrix = np.array([[data[key]] for key in keys])
        elif isinstance(data, np.matrix):
            self.matrix = np.array(data)
        elif isinstance(data, np.ndarray):
            self.matrix = data
        else:
            raise ProgModelTypeError(f"Input must be a dictionary or numpy array, not {type(data)}")     

    def __getitem__(self, key):
        return self.matrix[self._keys.index(key)][0].item()

    def __setitem__(self, key, value):
        self.matrix[self._keys.index(key)] = np.atleast_1d(value)

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    def __eq__(self, other):
        return self._keys == other._keys and (self.matrix == other.matrix).all()

    def __hash__(self):
        return hash(self.keys) + hash(self.matrix)
    
    def __str__(self):
        return self.__repr__()

    def keys(self):
        return self._keys

    def values(self):
        return np.array([value[0] for value in self.matrix])

    def items(self):
        return zip(self._keys, np.array([value[0] for value in self.matrix]))

    def __contains__(self, key):
        return key in self._keys

    def __repr__(self) -> str:
        return str({key: value[0] for key, value in zip(self._keys, self.matrix)})
