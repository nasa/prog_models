from numpy import float64

from prog_models.utils.containers import DictLikeMatrixWrapper
from prog_models.composite_model import CompositeModel

import pandas as pd
import numpy as np

df = DictLikeMatrixWrapper(['a', 'b', 'c'], {'a': 3, 'b': 1, 'c': 7})
df1 = df.copy()
print(df)
df = df.data.drop(['a', 'b', 'c'], axis=1)
print(df)
print(pd.DataFrame())

