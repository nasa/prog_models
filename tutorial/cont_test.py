from array import array

from prog_models.utils.containers import DictLikeMatrixWrapper
from prog_models.composite_model import CompositeModel

import pandas as pd
import numpy as np

"""
key_list = ['a', 'b', 'c']
np_arr = np.array([[1], [4], [2]])
np_matrix = np.matrix([[1], [4], [2]])
dict_test = {'a': 1, 'b': 4, 'c': 2}
con_dict = DictLikeMatrixWrapper(key_list, dict_test)

con_matrix = DictLikeMatrixWrapper(key_list, np_matrix)

con_array = DictLikeMatrixWrapper(key_list, np_arr)

print(con_dict)
print(con_matrix)
print(con_array)

for x in con_dict.items():
    print(x)
print('\n', con_dict.__len__())
print(con_matrix)
print(con_array.__eq__(con_dict))
test_copy = con_array.copy()
test_copy.update(DictLikeMatrixWrapper(['b'], {'b': 9}))
print(test_copy.__contains__('f'))
print(test_copy.__contains__('a'))

print('matrix:')
print(con_dict.matrix)
print(con_matrix.matrix)
print(con_array.matrix)

"""
# print(con_dict.data)
# print(con_matrix.data)

list_ky = ['a', 'b', 'c', 'd', 't']
dict_lin = {'a': 1, 'b': 4, 'c': 2}
# dict_none = DictLikeMatrixWrapper(list_ky, dict_lin)
# print(dict_none.data)

multi_dict = {'a': np.array([1., 2., 3., 4., 4.5]), 'b': np.array([5., 5., 5., 5., 5.]),
              'c': np.array([-3.2, -7.4, -11.6, -15.8, -17.9]), 't': np.array([0., 0.5, 1., 1.5, 2.])}
tester = DictLikeMatrixWrapper(list_ky, multi_dict)
print(tester)
# print(pd.DataFrame(multi_dict, columns=list_ky))
