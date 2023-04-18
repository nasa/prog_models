from array import array

from prog_models.utils.containers import DictLikeMatrixWrapper
from prog_models.composite_model import CompositeModel

import pandas as pd
import numpy as np

"""key_list = ['a', 'b', 'c']
np_arr = np.array([[1], [4], [2]])
np_matrix = np.matrix([[1], [4], [2]])
dict_test = {'a': 1, 'b': 4, 'c': 2}
con_dict = DictLikeMatrixWrapper(key_list, dict_test)

con_matrix = DictLikeMatrixWrapper(key_list, np_matrix)

con_array = DictLikeMatrixWrapper(key_list, np_arr)"""

# print(np.array_equal(con_array.matrix, con_matrix.matrix, equal_nan=False))
"""print(con_dict.matrix)
print(con_matrix.matrix)
print(con_array.matrix)"""
x_arr = np.array([[[1, 2, 3]], [[1, 3, 1]], [[4, 6, 2]]], dtype=np.float64)
dict_data = {'a': np.array([1, 2, 3]), 'b': np.array([3, 4, 3]), 'c': np.array([8, 2, 0])}
dlmw = DictLikeMatrixWrapper(['a', 'b', 'c'], dict_data)
print(dlmw.data)
print(dlmw.matrix)  # True

#print(len(list(dict_data.values())[0]))
