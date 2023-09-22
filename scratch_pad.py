import numpy as np

A = np.array([
    [2, 9, -3],
    [1, 2, 7],
    [1, 2, 3]
])

inv_A = np.linalg.inv(A)

print(inv_A)
# [[-2.  1.]
#  [ 5. -2.]]

I = inv_A @ A
print(I)
# [[1. 0.]
#  [0. 1.]]