import torch

t1 = torch.zeros(2, 1, 2, 1, 2) # 2x1x2x1x2
print(t1)
print(t1.size()) # 2x1x2x1x2
print(t1.squeeze())
print(t1.squeeze().size()) # 2x2x2
print(torch.squeeze(t1, dim=1).size()) # 2x2x1x2 (1차원에 해당하는 부분만 압축)
print()

t2 = torch.zeros(2,3)
print(t2.size()) # 2x3
print(torch.unsqueeze(t2, dim=0).size()) # 1x2x3 (0차원에 추가)
print(torch.unsqueeze(t2, dim=1).size()) # 2x1x3 (1차원에 추가)
print(torch.unsqueeze(t2, dim=2).size()) # 2x3x1 (2차원에 추가)
print()

print(torch.unsqueeze(t2, dim=0)) # 1x2x3 (0차원에 추가)
print(torch.unsqueeze(t2, dim=1)) # 2x1x3 (1차원에 추가)
print(torch.unsqueeze(t2, dim=2)) # 2x3x1 (2차원에 추가)
print()

import torch.nn.init as init

t3 = init.uniform_(torch.FloatTensor(3,4))
print(t3)
print()

t4 = init.normal_(torch.FloatTensor(3,4))
print(t4)
print()

t4 = init.normal_(torch.FloatTensor(3,4), mean=10, std=4)
print(t4)
print()

t5 = torch.FloatTensor(torch.randn(3,4)) # 표준정규분포
print(t5)
print()

t6 = init.constant_(torch.FloatTensor(3,4), 100)
print(t6)
print()

w = torch.tensor(2., requires_grad=True) # requires_grad=True 옵션이 있어야 미분 가능
y = 10 * w
y.backward()
print(w.grad)

t7 = torch.tensor(3., requires_grad=True)
for _ in range(20):
    y = 7 * t7
    y.backward()
    print('gradient: ', t7.grad)
    t7.grad.zero_() # 초기화를 하지 않으면 값이 누적됨