# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

import numpy as np
from typing import Union

from ..exceptions import ProgModelTypeError


class DictLikeMatrixWrapper():
    """
    A container that behaves like a dictionary, but is backed by a numpy array, which is itself directly accessable. This is used for model states, inputs, and outputs- and enables efficient matrix operations.

    Arguments
    ---------
    keys: list
        The keys of the dictionary. e.g., model.states or model.inputs
    data: dict or numpy array
        The contained data (e.g., :term:`input`, :term:`state`, :term:`output`). If numpy array should be column vector in same order as keys
    """

    def __init__(self, keys: list, data: Union[dict, np.array]):
        if not isinstance(keys, list):  # if keys is not a list- try to convert it to one
            keys = list(keys)
        self._keys = keys.copy()  # saves list of keys in object, ids for data
        if isinstance(data, np.matrix):
            self.matrix = np.array(data, dtype=np.float64)
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                data = data[np.newaxis].T
            self.matrix = data
        elif isinstance(data, (dict, DictLikeMatrixWrapper)):
            self.matrix = np.array(
                [
                    [data[key]] if key in data else [None] for key in keys
                ], dtype=np.float64)
        else:
            raise ProgModelTypeError(f"Data must be a dictionary or numpy array, not {type(data)}")

    def __reduce__(self):
        return (DictLikeMatrixWrapper, (self._keys, self.matrix))

    # get all values associated with a key, ex: all values of 'i'
    def __getitem__(self, key: str) -> int:
        row = self.matrix[self._keys.index(key)]  # creates list from a row of matrix
        if len(row) == 1:  # if row list contains 1 value, returns that value (non-vectorized)
            return row[0]
        return row  # else returns entire row/list (vectorized case)

    # sets a row at the key given
    def __setitem__(self, key: str, value: int) -> None:
        index = self._keys.index(key)  # the int value index for the key given
        self.matrix[index] = np.atleast_1d(value)

    # removes row associated with key
    def __delitem__(self, key: str) -> None:
        self.matrix = np.delete(self.matrix, self._keys.index(key), axis=0)
        self._keys.remove(key)

    # add another matrix to the existing matrix
    def __add__(self, other: "DictLikeMatrixWrapper") -> "DictLikeMatrixWrapper":
        return DictLikeMatrixWrapper(self._keys, self.matrix + other.matrix)

    # creates iterator object for the list of keys
    def __iter__(self):
        return iter(self._keys)

    # returns the length of key list
    def __len__(self) -> int:
        return len(self._keys)

    # Compares two DictLikeMatrixWrappers (i.e. *Containers) or a DictLikeMatrixWrapper and a dictionary
    def __eq__(self, other: "DictLikeMatrixWrapper") -> bool:
        if isinstance(other, dict):
            # checks that the list of keys for each matrix match
            list_key_check = (list(self.keys()) == list(other.keys()))
            # checks to see that each row matches
            matrix_check = (self.matrix == np.array([[other[key]] for key in self._keys])).all()
            return list_key_check and matrix_check
        
        # Case where other is a DictLikeMatrixWrapper as well
        list_key_check = self.keys() == other.keys()
        matrix_check = (self.matrix == other.matrix).all()
        return list_key_check and matrix_check

    # returns hash value sum for keys and matrix
    def __hash__(self):
        return hash(self.keys) + hash(self.matrix)

    # (?) why
    def __str__(self) -> str:
        return self.__repr__()

    # gets the list of values associated with the key given
    def get(self, key, default=None):
        if key in self._keys:
            return self[key]
        return default

    # creates copy of object
    def copy(self) -> "DictLikeMatrixWrapper":
        return DictLikeMatrixWrapper(self._keys, self.matrix.copy())

    # returns list of keys
    def keys(self) -> list:
        return self._keys

    # returns array of matrix values
    def values(self) -> np.array:
        # if the first row of the matrix has one value
        if len(self.matrix) > 0 and len(self.matrix[0]) == 1:
            return np.array([value[0] for value in self.matrix])  # returns the value from the first row
        # else returns the matrix (vectorized case)
        return self.matrix

    # returns keys and values as a list of tuples (for iterating)
    def items(self) -> zip:
        # if the first row of the matrix has one value (non-vectorized case)
        if len(self.matrix) > 0 and len(self.matrix[0]) == 1:
            return zip(self._keys, np.array([value[0] for value in self.matrix]))
        return zip(self._keys, self.matrix)

    # update values by merging in other DictLikeMatrixWrapper
    def update(self, other: "DictLikeMatrixWrapper") -> None:
        for key in other.keys():
            if key in self._keys:
                # Existing key
                self[key] = other[key]
            else:
                # A new key!
                self._keys.append(key)
                self.matrix = np.vstack((self.matrix, np.array([other[key]])))

    # boolean showing whether the key exists
    def __contains__(self, key: str) -> bool:
        return key in self._keys

    # returns a string of dictionaries, tuples from keys and values
    def __repr__(self) -> str:
        # if the matrix has rows and first row/list has one value in it
        if len(self.matrix) > 0 and len(self.matrix[0]) == 1:
            # returns the key and associated value
            return str({key: value[0] for key, value in zip(self._keys, self.matrix)})
        # else returns a string of dictionaries containing all the keys and associated matrix values
        return str(dict(zip(self._keys, self.matrix)))