# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from numpy import float64, matrix, ndarray, array, newaxis, nan, delete, atleast_1d, vstack
import pandas as pd
from prog_models.exceptions import ProgModelTypeError
from typing import Union


class DictLikeMatrixWrapper():
    """
    A container that behaves like a dictionary, but is backed by a numpy array, which is itself directly accessable. This is used for model states, inputs, and outputs- and enables efficient matrix operations.

    Arguments:
        keys -- list: The keys of the dictionary. e.g., model.states or model.inputs
        data -- dict or numpy array: The contained data (e.g., :term:`input`, :term:`state`, :term:`output`). If numpy array should be column vector in same order as keys
    """

    def __init__(self, keys: list, data: Union[dict, array]):
        """
        Initializes the container
        """
        if not isinstance(keys, list):
            keys = list(keys)  # creates list with keys
        self._keys = keys.copy()
        if isinstance(data, matrix):
            self.data = pd.DataFrame(array(data, dtype=float64), self._keys, dtype=float64)
            self.matrix = self.data.to_numpy(dtype=float64)
        elif isinstance(data, ndarray):
            if data.ndim == 1:
                data = data[newaxis].T
                self.data = pd.DataFrame(data, self._keys)
            self.data = pd.DataFrame(data, self._keys).T
            self.matrix = data
        elif isinstance(data, (dict, DictLikeMatrixWrapper)):
            if data and not isinstance(list(data.values())[0], ndarray):  # len(self.matrix[0]) == 1:
                if isinstance(data, DictLikeMatrixWrapper):
                    data = dict(data.copy())
                self.data = pd.DataFrame(data, columns=self._keys, index=[0], dtype=float64).astype(object).replace(
                    nan, None)
            else:
                self.data = pd.DataFrame(data, columns=self._keys)
            self.matrix = self.data.to_numpy(dtype=float64).T if len(data) > 0 else array([])
        else:
            raise ProgModelTypeError(f"Data must be a dictionary or numpy array, not {type(data)}")

    def __reduce__(self):
        """
        reduce is overridden for pickles
        """
        return DictLikeMatrixWrapper, (self._keys, self.matrix)

    def getmatrix(self) -> ndarray:
        """
        creates a numpy array corresponding to the pd.DataFrame data

        Returns: np.ndarray
        """
        matrix_np = self.data.to_numpy().T
        return matrix_np

    def __getitem__(self, key: str) -> int:
        """
        get all values associated with a key, ex: all values of 'i'
        """
        # row = self.data.loc[:, key].to_list()  # creates list from a row of the DataFrame data
        row = self.matrix[self._keys.index(key)]  # creates list from a row of matrix
        if len(row) == 1:  # list contains 1 value, returns that value (non-vectorized)
            return row[0]
        else:
            return row  # returns entire row/list (vectorized case)

    def __setitem__(self, key: str, value: int) -> None:
        """
        sets a row at the key given
        """
        index = self._keys.index(key)  # the int value index for the key given
        self.matrix[index] = atleast_1d(value)

    def __delitem__(self, key: str) -> None:
        """
        removes row associated with key
        """
        self.matrix = delete(self.matrix, self._keys.index(key), axis=0)
        self._keys.remove(key)

    def __add__(self, other: "DictLikeMatrixWrapper") -> "DictLikeMatrixWrapper":
        """
        add another matrix to the existing matrix
        """
        return DictLikeMatrixWrapper(self._keys, self.matrix + other.matrix)

    def __iter__(self):
        """
        creates iterator object for the list of keys
        """
        return iter(self._keys)

    def __len__(self) -> int:
        """
        returns the length of key list
        """
        return len(self._keys)

    def __eq__(self, other: "DictLikeMatrixWrapper") -> bool:
        """
        Compares two DictLikeMatrixWrappers (i.e. *Containers) or a DictLikeMatrixWrapper and a dictionary
        """
        if isinstance(other, dict):  # checks that the list of keys for each matrix match
            list_key_check = (list(self.keys()) == list(
                other.keys()))  # checks that the list of keys for each matrix are equal
            matrix_check = (self.matrix == array(
                [[other[key]] for key in self._keys])).all()  # checks to see that each row matches
            return list_key_check and matrix_check
        list_key_check = self.keys() == other.keys()
        matrix_check = (self.matrix == other.matrix).all()
        return list_key_check and matrix_check

    def __hash__(self):
        """
        returns hash value sum for keys and matrix
        """
        return hash(self.keys) + hash(self.matrix)

    def __str__(self) -> str:
        """
        Represents object as string
        """
        return self.__repr__()

    def get(self, key, default=None):
        """
        gets the list of values associated with the key given
        """
        if key in self._keys:
            return self[key]
        return default

    def copy(self) -> "DictLikeMatrixWrapper":
        """
        creates copy of object
        """
        return DictLikeMatrixWrapper(self._keys, self.matrix.copy())

    def keys(self) -> list:
        """
        returns list of keys for container
        """
        return self._keys

    def values(self) -> array:
        """
        returns array of matrix values
        """
        if len(self.matrix) > 0 and len(
                self.matrix[0]) == 1:  # if the first row of the matrix has one value (i.e., non-vectorized)
            return array([value[0] for value in self.matrix])  # the value from the first row
        return self.matrix  # the matrix (vectorized case)

    def items(self) -> zip:
        """
        returns keys and values as a list of tuples (for iterating)
        """
        if len(self.matrix) > 0 and len(
                self.matrix[0]) == 1:  # first row of the matrix has one value (non-vectorized case)
            return zip(self._keys, array([value[0] for value in self.matrix]))
        return zip(self._keys, self.matrix)

    def update(self, other: "DictLikeMatrixWrapper") -> None:
        """
        merges other DictLikeMatrixWrapper, updating values
        """
        for key in other.keys():
            if key in self._keys:  # checks to see if every key in 'other' is in 'self'
                # Existing key
                self[key] = other[key]
            else:  # else it isn't it is appended to self._keys list
                # A new key!
                self._keys.append(key)
                self.matrix = vstack((self.matrix, array([other[key]])))

    def __contains__(self, key: str) -> bool:
        """
        boolean showing whether the key exists

        example
        -------
        >>> from prog_models.utils.containers import DictLikeMatrixWrapper
        >>> dlmw = DictLikeMatrixWrapper(['a', 'b', 'c'], {'a': 1, 'b': 2, 'c': 3})
        >>> 'a' in dlmw  # True
        """
        return key in self._keys

    def __repr__(self) -> str:
        """
        represents object as string

        returns: a string of dictionaries containing all the keys and associated matrix values
        """
        if len(self.matrix) > 0 and len(
                self.matrix[0]) == 1:  # the matrix has rows and the first row/list has one value in it
            return str({key: value[0] for key, value in zip(self._keys, self.matrix)})
        return str(dict(zip(self._keys, self.matrix)))
