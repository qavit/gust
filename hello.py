import numpy as np
import pandas as pd

# Python course 2024-08-12

# Practice 1

X = np.random.randint(-10, 10, size=(2, 1))
W = np.random.randint(-10, 10, size=(3, 2))
B = np.random.randint(-10, 10, size=(3, 1))
Z = np.matmul(W, X) + B
Y = np.maximum(Z, 0)

# print(X)
# print(W)
# print(Z)
# print(Y)

# Practice 2

companies = [f'Company {chr(a)}' for a in range(65, 65+4)]
revenues = [[12000, 7000, 9000, 10000],
            [11000, 12000, 8000, 6000],
            [8000, 13000, 6000, 7500],
            [9000, 10000, 15000, 12000]]
quaters = [f'Q{i}' for i in range(1, 1+4)]

di = {c: v for c, v in zip(companies, revenues)}

df = pd.DataFrame(data=di, index=quaters)

print(df)
print(df.describe())
