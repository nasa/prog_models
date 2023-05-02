from numpy import float64

from prog_models.utils.containers import DictLikeMatrixWrapper
from prog_models.composite_model import CompositeModel

import pandas as pd
import numpy as np

"""df = DictLikeMatrixWrapper(['a', 'b', 'c'], {'a': 3, 'b': 1, 'c': 7})
df1 = df.copy()
print(df)
arr_df = []
i = 10
while i > 0:
    arr_df.append(df)
    i = i-1
# print(arr_df)
data_df = []
print(df.data)
df.data = df.data.drop(index=0)
print(df.data)"""

state = [{'a': i * 25, 'b': i * 50} for i in range(10)]
state2 = [{'a': 0, 'b': 0}, {'a': 25, 'b': 50}, {'a': 50, 'b': 100}, {'a': 75, 'b': 150},
                                        {'a': 100, 'b': 200}, {'a': 125, 'b': 250}, {'a': 150, 'b': 300},
                                        {'a': 175, 'b': 350}, {'a': 200, 'b': 400}, {'a': 225, 'b': 450}]
print(state2,'\n', state)

