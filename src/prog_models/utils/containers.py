# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

import numpy as np
from typing import Union

from prog_models.exceptions import ProgModelTypeError


class DictLikeMatrixWrapper():
    """
    A container that behaves like a dictionary, but is backed by a numpy array, which is itself directly accessable. This is used for model states, inputs, and outputs- and enables efficient matrix operations.
    
    Arguments:
        keys -- list: The keys of the dictionary. e.g., model.states or model.inputs
        data -- dict or numpy array: The contained data (e.g., :term:`input`, :term:`state`, :term:`output`). If numpy array should be column vector in same order as keys
    """

    def __init__(self, keys: list, data: Union[dict, np.array]):
        """ Initializes the container
        """
        if not isinstance(keys, list):
            keys = list(keys)   # creates list with keys
        self._keys = keys.copy()
        if isinstance(data, np.matrix):
            self.matrix = np.array(data, dtype=np.float64)
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                data = data[np.newaxis].T
            self.matrix = data
        elif isinstance(data, (dict, DictLikeMatrixWrapper)):
            if not data or not isinstance(list(data.values())[0], np.ndarray):
                # matrix structure - data is an empty dict or dict with a single value each
                self.matrix = np.array(
                    [
                        [data[key]] if key in data else [None] for key in keys
                    ], dtype=np.float64)
            else:
                # matrix structure - when data is a dict with arrays as values
                self.matrix = np.array(
                    [
                        data[key] for key in keys if key in data
                    ], dtype=np.float64)
        else:
            raise ProgModelTypeError(f"Data must be a dictionary or numpy array, not {type(data)}")

    def __reduce__(self):
        """
        reduce is overridden for pickles
        """
        return (DictLikeMatrixWrapper, (self._keys, self.matrix))

    def __getitem__(self, key: str) -> int:
        """
        get all values associated with a key, ex: all values of 'i'
        """
        row = self.matrix[self._keys.index(key)]    #creates list from a row of matrix
        if len(row) == 1:   #list contains 1 value, returns that value (non-vectorized)
            return row[0]
        return row #returns entire row/list (vectorized case)

    def __setitem__(self, key: str, value: int) -> None:
        """
        sets a row at the key given
        """
        index = self._keys.index(key)   #the int value index for the key given
        self.matrix[index] = np.atleast_1d(value)

    
    def __delitem__(self, key: str) -> None:
        """
        removes row associated with key
        """
        self.matrix = np.delete(self.matrix, self._keys.index(key), axis=0)
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
        if isinstance(other, dict): #checks that the list of keys for each matrix match
            list_key_check = (list(self.keys()) == list(other.keys()))  #checks that the list of keys for each matrix are equal
            matrix_check = (self.matrix == np.array([[other[key]] for key in self._keys])).all()    #checks to see that each row matches
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

    def values(self) -> np.array:
        """
        returns array of matrix values
        """
        if len(self.matrix) > 0 and len(self.matrix[0]) == 1:   #if the first row of the matrix has one value (i.e., non-vectorized)
            return np.array([value[0] for value in self.matrix])    #the value from the first row
        return self.matrix    # the matrix (vectorized case)

    def items(self) -> zip:
        """
        returns keys and values as a list of tuples (for iterating)
        """
        if len(self.matrix) > 0 and len(self.matrix[0]) == 1:   #first row of the matrix has one value (non-vectorized case)
            return zip(self._keys, np.array([value[0] for value in self.matrix]))
        return zip(self._keys, self.matrix)

    def update(self, other: "DictLikeMatrixWrapper") -> None:
        """
        merges other DictLikeMatrixWrapper, updating values
        """
        for key in other.keys():   
            if key in self._keys:   #checks to see if every key in 'other' is in 'self'
                # Existing key
                self[key] = other[key]
            else:   #else it isn't it is appended to self._keys list
                # A new key!
                self._keys.append(key)
                self.matrix = np.vstack((self.matrix, np.array([other[key]])))

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
        if len(self.matrix) > 0 and len(self.matrix[0]) == 1:   #the matrix has rows and the first row/list has one value in it
            return str({key: value[0] for key, value in zip(self._keys, self.matrix)})
        return str(dict(zip(self._keys, self.matrix)))  
