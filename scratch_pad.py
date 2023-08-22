import numpy as np

A = np.array([[2, 1],
              [-1, -2]])

A_inv = np.linalg.inv(A)
B = np.array([5,-4])

print(A_inv @ B) # [2. 1.]