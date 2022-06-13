import torch

t1 = torch.tensor([1, 2, 3, 4, 5, 6]).view(3, 2)
t2 = torch.tensor([7, 8, 9, 10, 11, 12]).view(2, 3)
print(t1)
print()
print(t2)
print()

print(torch.mm(t1, t2))
print()
print(torch.matmul(t1, t2))
print()

t4 = torch.FloatTensor(2, 4, 3) # 2x4x3
print(t4)
print()
t5 = torch.FloatTensor(2, 3, 5) # 2x3x5
print(t5)
print()
print(torch.bmm(t4, t5)) # 2x4x5 (batch 형태로 4x3과 3x5를 곱한 행렬)