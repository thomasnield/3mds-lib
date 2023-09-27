import numpy as np
from numpy.linalg import inv

x = np.array([84,37,58,52,47,78,93,15,12,60])
y = np.array([155.8,102.0,164.8,120.9,86.8,93.0,201.6,25.2,14.7,118.6])

x_1 = np.vstack([x, np.ones(len(x))]).T

coeffs = inv(x_1.transpose() @ x_1) @ (x_1.transpose() @ y)

print(coeffs)
