import pandas as pd
import numpy as np

# example from:
# Linear Algebra and its applications 5th edition, David Lay
# Example 1 a)
# pg 35


# numpy version
A = np.matrix([[1, 2, -1], [0, -5, 3]]) # matrix
x = np.array([[4], [3], [7]])   # column vector
result = np.dot(A, x)   # matrix product mult.
print(result)   # should be [[3], [6]] a column vector

A_df = pd.DataFrame([[1, 2, -1], [0, -5, 3]])   # matrix A
x_df = pd.DataFrame([[4], [3], [7]])    # column vector x
result_df = A_df.dot(x_df)  # Ax
print(result_df)    # same result as above


dict_lin = {'a': 1, 'b': 4, 'c': 2}
num = len(list(dict_lin.values()))
print(type(list(dict_lin.values())[0]))

