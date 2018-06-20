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
print(np.sum(H, 2).shape)
print(np.tensordot(H, A, axes=((3), (1))).shape)
H = np.insert(H, 0, 0, axis=-1)
# print(E)
# print(H)
print(np.diff(H))
A = A.reshape(-1, 1)
C = A == B
C = C.astype(int)
print((np.random.dirichlet(np.ones(3) / 3)))
