import torch
import torch.nn.functional as F

torch.manual_seed(777)

x1 = torch.randn(3, 5, requires_grad=True) # 1x5의 벡터가 3개. x1을 최종 레이어(5개 노드)의 출력값이라 가정
hypothesis = F.softmax(x1, dim=1) # dim=1은 1x5짜리 벡터에 대해 softmax를 계산함을 의미
print(x1)
print(hypothesis)

y = torch.randint(5, (3,)).long() # 0~5 사이의 값 3개
print(y)
y_one_hot = torch.zeros_like(hypothesis)
print(y_one_hot)
# print(y.unsqueeze(1))
y_one_hot.scatter_(1, y.unsqueeze(1), 1) # dim 1 기준으로 y.unsqueeze(1) 위치에 1을 넣음. (1, 3), (1, 0), (1, 0)에 1을 넣음
print()
print(y_one_hot)
print()

# 아래 세 가지는 모두 동일함
print((-y_one_hot * torch.log(F.softmax(x1, dim=1))).sum(dim=1).mean()) # loss 계산
print((-y_one_hot * F.log_softmax(x1, dim=1)).sum(dim=1).mean()) # F.log_softmax(x1) = torch.log(F.softmax(x1))
print(F.cross_entropy(x1, y)) # y가 정수값 그대로 들어감. 내부적으로 one-hot encoding을 자동으로 수행함