import numpy as np

X = np.random.randint(-10, 10, size=(2, 1))
W = np.random.randint(-10, 10, size=(3, 2))
B = np.random.randint(-10, 10, size=(3, 1))
Z = np.matmul(W, X) + B
Y = np.maximum(Z, 0)

print(X)
print(W)
print(Z)
print(Y)
