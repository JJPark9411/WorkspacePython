import torch

t1 = torch.tensor([1, 2, 3, 4, 5, 6])
print(t1)
print(t1.size())
t2 = t1.view(2,3)
print(t2)
print()

t3 = torch.tensor([[1, 2], [3, 4], [5, 6]])
print(t3)
print()
print(t3.view(-1)) # 0차원에 직렬화 print(t3.view(0, -1))
print(t3.view(1, -1)) # 1차원에 직렬화
print()

t4 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]) # 2x2x2 [0차원]x[1차원]x[2차원]
print(t4)
print()

print(t4.view(-1)) # 0차원 직렬화
print(t4.view(1, -1)) # 1차원 직렬화
print(t4.view(2, -1)) # 2차원 직렬화