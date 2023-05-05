from typing import Union

from numpy import float64

from prog_models.utils.containers import DictLikeMatrixWrapper
from prog_models.composite_model import CompositeModel
from prog_models.sim_result import SimResult, LazySimResult

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

# Variables
def f(x):
    return {k: v * 2 for k, v in x.items()}

time = list(range(5))
state = [{'a': i * 2.5, 'b': i * 5} for i in range(5)]
result = LazySimResult(f, time, state)
"""column_val = list(zip(*[['state']*len(state[0]), state[0].keys()]))
index = pd.MultiIndex.from_tuples(column_val)
frame = pd.DataFrame(data=state, columns=index)"""

time = list(range(10))  # list of int, 0 to 9
state = [{'a': i * 2.5, 'b': i * 5} for i in range(10)]
result2 = SimResult(time, state)
test = result2.to_numpy()
print('resut2: ', result2.monotonicity(), result2)
inputs_plus = [{'i1': 1, 'i2': 2.1}, {'i1': 1.0, 'i2': 2.1}, {'i1': 1.0, 'i2': 2.1}, {'i1': 1.0, 'i2': 2.1}, {'i1': 1.0, 'i2': 2.1}, {'i1': 1.0, 'i2': 2.1}, {'i1': 1.0, 'i2': 2.1}]
inputs = [{'i1': 1, 'i2': 2.1}, {'i1': 1.0, 'i2': 2.1}, {'i1': 1.0, 'i2': 2.1}, {'i1': 1.0, 'i2': 2.1}, {'i1': 1.0, 'i2': 2.1}, {'i1': 1.0, 'i2': 2.1}, {'i1': 1.0, 'i2': 2.1}]
print(np.array_equal(inputs, inputs_plus))

"""result = []  # list for dataframes of monotonicity values
for label in result.frame.columns:  # iterates through each column label
    mono_sum = 0
    for i in list(len(result.frame.index)):  # iterates through for calculating monotonocity
        # print(test_df[label].iloc[i+1], ' - ', test_df[label].iloc[i])
        mono_sum += np.sign(result.frame[label].iloc[i + 1] - result.frame[label].iloc[i])
    result.append(pd.DataFrame({label: abs(mono_sum / (len(result.frame.index) - 1))},
                               index=['monotonicity']))  # adds each dataframe to a list
temp_pd = pd.concat(result, axis=1)
print(temp_pd.drop(columns=['time']))"""


