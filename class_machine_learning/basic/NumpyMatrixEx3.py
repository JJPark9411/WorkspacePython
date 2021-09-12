import numpy as np

A = np.array([
    [1, -1, 2],
    [3, 2, 2]
])

B = np.array([
    [0, 1],
    [-1, 1],
    [5, 2]
])

C = np.array([
    [2, -1],
    [-3, 3]
])

D = np.array([
    [-5, 1],
    [2, 0]
])

# 행렬 연산을 result 변수에 저장하세요
result = B.T @ (2*A.T) @ (3*np.linalg.pinv(C) + D.T)

print(result)