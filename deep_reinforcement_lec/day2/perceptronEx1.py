import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(777) # seed를 고정해서 초기 weights가 동일하게 나오도록 함

x = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
y = torch.FloatTensor([[0], [1], [1], [1]]) # OR
# y = torch.FloatTensor([[0], [1], [1], [0]]) # XOR -> 단층 퍼셉트론으로 학습이 되지 않음

model = nn.Sequential(
    nn.Linear(2, 1, bias=True), # default: bias=True
    nn.Sigmoid()

    # nn.Linear(2, 2), # 다층 퍼셉트론이므로 XOR도 학습 가능
    # nn.Sigmoid(),
    # nn.Linear(2, 1),
    # nn.Sigmoid()
)

loss_func = nn.BCELoss() # F.binary_cross_entropy()와 동일
optimizer = optim.SGD(model.parameters(), lr=1)

for epoch in range(10001):
    hypothesis = model(x)
    optimizer.zero_grad()
    loss = loss_func(hypothesis, y)
    loss.backward()
    optimizer.step()

    if epoch%100 == 0:
        print('epoch: {}, loss: {:.4f}'.format(epoch, loss.item()))

with torch.no_grad(): # 미분하지 않도록 설정 -> 연산량 감소에 도움이 됨
    hypothesis = model(x)
    prediction = (hypothesis > 0.5).float()
    accuracy = (prediction == y).float().mean()
    print('hypothesis:\n{}\nprediction:\n{}\naccuracy: {}'.format(
        hypothesis.numpy(), prediction.numpy(), accuracy.item()
        # hypothesis, prediction, accuracy.item()
    ))