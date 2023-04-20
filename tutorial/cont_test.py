

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
dict_data = {'a': np.array([1]), 'b': np.array([3]), 'c': np.array([8])}
dlmw = DictLikeMatrixWrapper(['a', 'b', 'c'], dict_data)
print(dlmw.data.loc[:,'a'].to_list())
row = dlmw.data.loc[:,'a'].to_list()  # creates list from a row of the DataFrame data
if len(row) == 1:  # list contains 1 value, returns that value (non-vectorized)
    print(dlmw.data.loc[0, 'a'])
else:
    print(row)  # returns entire row/list (vectorized case)
