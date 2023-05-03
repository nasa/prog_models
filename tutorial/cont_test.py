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

time = list(range(5))
state = [{'a': i * 2.5, 'b': i * 5} for i in range(5)]
frame = pd.DataFrame(state)
frame.insert(0, "time", time)
result = SimResult(time, state)
list(result)
print(list(result), '\n', state)
print(time, '\n', result.times[0])
print(frame.equals(result.frame))

# print(temp_df.loc[isinstance(temp_df.isin(d_dict).iat[0], Union[int, float])])
