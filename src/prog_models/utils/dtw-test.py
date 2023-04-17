import numpy as np

def helperDTW(s, t):
    n, m = len(s), len(t)
    dtw_matrix = np.zeros((n+1, m+1))
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(s[i-1][0] - t[j-1][0])
            # take last min from a square box
            last_min = np.min([dtw_matrix[i-1, j],
                            dtw_matrix[i, j-1], 
                            dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix[n, m]

a = [[1], [2], [3]]
b = [[2], [2], [2], [3], [4]]

x = helperDTW(a, b)
print(x)