# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

import numpy as np
from typing import Union
import pandas as pd

from prog_models.exceptions import ProgModelTypeError


class DictLikeMatrixWrapper():
    """
    A container that behaves like a dictionary, but is backed by a numpy array, which is itself directly accessable. This is used for model states, inputs, and outputs- and enables efficient matrix operations.

    Arguments:
        keys -- list: The keys of the dictionary. e.g., model.states or model.inputs
        data -- dict or numpy array: The contained data (e.g., :term:`input`, :term:`state`, :term:`output`). If numpy array should be column vector in same order as keys
    """

    def __init__(self, keys: list, data: Union[dict, np.array, pd.Series]):
        """
        Initializes the container
        """
        if not isinstance(keys, list):
            keys = list(keys)  # creates list with keys
        temp_keys = keys.copy()
        if isinstance(data, np.matrix):
            self.data = pd.DataFrame(np.array(data, dtype=np.float64), temp_keys)
        elif isinstance(data, np.ndarray):  # data is a multidimensional array, in column vector form
            self.data = pd.DataFrame(data, temp_keys)
        elif isinstance(data, (dict, DictLikeMatrixWrapper)):  # data is not in column vector form
            self.data = pd.DataFrame(data, index=[0]).T
        else:
            raise ProgModelTypeError(f"Data must be a dictionary or numpy array, not {type(data)}")
        self.matrix = self.data.to_numpy()
        self._keys = self.data.index.to_list()
    def __reduce__(self):
        """
        reduce is overridden for pickles
        """
        keys = self.data.index.to_list()
        matrix = self.data.to_numpy()
        return (DictLikeMatrixWrapper, (keys, matrix))
    def __getitem__(self, key: str) -> int:
        """
        get all values associated with a key, ex: all values of 'i'
        """
        row = self.data.loc[key].to_list()  # creates list from a row of the DataFrame data
        if len(self.data.loc[key]) == 1:  # list contains 1 value, returns that value (non-vectorized)
            return self.data.loc[key, 0]
        return row  # returns entire row/list (vectorized case)

    def __setitem__(self, key: str, value: int) -> None:
        """
        sets a row at the key given
        """
        self.data.loc[key] = np.atleast_1d(value)  # using the key to find the Series location


    def __delitem__(self, key: str) -> None:
        """
        removes row associated with key
        """
        self.data = self.data.drop(index=[key])

    def __add__(self, other: "DictLikeMatrixWrapper") -> "DictLikeMatrixWrapper":
        """
        add 'other' matrix to the existing matrix
        """
        df_summed = self.data.add(other.data)  # the values in self and other summed in new series
        key_list = self.data.index.to_list()
        return DictLikeMatrixWrapper(key_list, df_summed.to_numpy())

    def __iter__(self):
        """
        creates iterator object for the list of keys
        """
        return iter(self.data.index.to_list())

    def __len__(self) -> int:
        """
        returns the length of key list
        """
        return len(self.data.index)

    def __eq__(self, other: "DictLikeMatrixWrapper") -> bool:
        """
        Compares two DictLikeMatrixWrappers (i.e. *Containers) or a DictLikeMatrixWrapper and a dictionary
        """
        if isinstance(other, dict):  # checks that the list of keys for each matrix match
            other_series = pd.Series(other)
            return self.data.equals(other_series)
        return self.data.equals(other.data)

    def __hash__(self):
        """
        returns hash value sum for keys and matrix
        """
        sum_hash = 0
        for x in pd.util.hash_pandas_object(self.data):
            sum_hash = sum_hash + x
        return sum_hash

    def __str__(self) -> str:
        """
        Represents object as string
        """
        return self.__repr__()

    def get(self, key, default=None):
        """
        gets the list of values associated with the key given
        """
        if key in self.data.index:
            return self.data.loc[key, 0]
        return default

    def copy(self) -> "DictLikeMatrixWrapper":
        """
        creates copy of object
        """
        return DictLikeMatrixWrapper(self._keys, self.matrix.copy())
        keys = self.data.index.to_list()
        matrix = self.data.to_numpy().copy()
        return DictLikeMatrixWrapper(keys, matrix)

    def keys(self) -> list:
        """
        returns list of keys for container
        """
        keys = self.data.index.to_list()
        return keys

    def values(self) -> np.array:
        """
        returns array of matrix values
        """
        matrix = self.data.to_numpy()
        return matrix

    def items(self) -> zip:
        """
        returns keys and values as a list of tuples (for iterating)
        """
        if len(self.data.index) > 0:  # first row of the matrix has one value (non-vectorized case)
            np_array = np.array([value[1] for value in self.data.items()])
            return zip(self.data.index.to_list(), np_array[0])
        return zip(self.data.index.to_list(), self.data.to_list())

    def update(self, other: "DictLikeMatrixWrapper") -> None:
        """
        merges other DictLikeMatrixWrapper, updating values
        """
        for key in other.data.index.to_list():
            if key in self.data.index.to_list():  # checks to see if the key exists
                # Existing key
                self.data.loc[key] = other.data.loc[key]
            else:  # the key doesn't exist within
                # the key
                temp_df = DictLikeMatrixWrapper([key], {key: other.data.loc[key, 0]})
                self.data = pd.concat([self.data, temp_df.data])

    def __contains__(self, key: str) -> bool:
        """
        boolean showing whether the key exists

        example
        -------
        >>> from prog_models.utils.containers import DictLikeMatrixWrapper
        >>> dlmw = DictLikeMatrixWrapper(['a', 'b', 'c'], {'a': 1, 'b': 2, 'c': 3})
        >>> 'a' in dlmw  # True
        """
        key_list = self.data.index.to_list()
        return key in key_list

    def __repr__(self) -> str:
        """
        represents object as string

        returns: a string of dictionaries containing all the keys and associated matrix values
        """
        return str(self.data.to_dict()[0])