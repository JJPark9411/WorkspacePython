import torch

t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(t1)
print()
print(t1[:, :2])
print()
print(t1[t1 > 4])

t1[:, 2] = 1000
print()
print(t1)
print()

t1[t1 > 4] = -50
print(t1)
print()

t2 = torch.tensor([[1, 2, 3], [4, 5, 6]]) # 2x3
t3 = torch.tensor([[7, 8, 9], [10, 11, 12]]) # 2x3
print(t2)
print(t3)
print()
t4 = torch.cat([t2, t3], dim=0) # 4x3
print(t4)
print()
print(torch.cat([t2, t3], dim=1)) # 2x6
print()

ch1 = torch.chunk(t4, 4, dim=0)
print(ch1)
print()

ch2 = torch.chunk(t4, 3, dim=1)
print(ch2)
print()