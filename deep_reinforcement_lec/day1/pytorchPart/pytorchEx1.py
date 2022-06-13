import torch

t1 = torch.FloatTensor([4, 5, 6, 7, 8])
print(t1) # tensor([4., 5., 6., 7., 8.])

print()
t2 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
print(t2)
print(t2.size())
print(t2.numpy()) # torch.tensor -> numpy array
print()

import numpy as np

ndata = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
t3 = torch.from_numpy(ndata) # numpy array -> torch.tensor
print(t3)
print()

t4 = torch.tensor([1, 2, 3])
t5 = torch.tensor([5, 6, 7])

t6 = t4 * 2
print(t6)

t7 = t5 - t4
print(t7)
print()

t8 = torch.tensor([[10, 20, 30], [40, 50, 60]]) # broadcasting 연산
print(t8 + t4)
print()

t9 = torch.linspace(0, 3, 20) # [0, 3] 구간의 샘플 20개
print(t9)
print()
print(torch.exp(t9))
print()

t10 = torch.tensor([[1, 4, 6], [1, 3, 9]])
print(t10)
print()

print(torch.max(t10, dim=0))
print()
print(torch.max(t10, dim=1))
print()
print("values:\t", torch.max(t10, dim=1)[0]) # values
print("indices:", torch.max(t10, dim=1)[1]) # indices