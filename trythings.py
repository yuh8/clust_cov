import numpy as np

A = np.array([[1, 2, 3], [6, 7, 8]])
B = np.array([3, 2, 3, 4, 5])
C = A.reshape(-1, 1) + B
C = C.reshape(2, 3, 5)
D = -np.arange(15)
D = D.reshape(3, 5)
E = C + D
K = np.array([0, 4, 1])
F = E.reshape(-1, 1)
G = F + K
H = G.reshape(2, 3, 5, 3)
print(H.shape)
print(H)
print(np.swapaxes(H, 1, 3))
