import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F

x = init.uniform(torch.Tensor(1000,1), -10, 10) # -10~10
value = init.normal_(torch.Tensor(1000,1), std=0.2)
y_target = 2*x + 3 + value # value는 noise 역할인 듯

model = nn.Linear(1,1) # input: 1 -> output: 1
optimizer = optim.SGD(model.parameters(), lr=0.01) # model.parameters(): weights and bias
# cost_func = nn.MSELoss() # cost func을 미리 지정하는 방법

for epoch in range(500):
    optimizer.zero_grad()
    hypothesis = model(x)
    cost = F.mse_loss(hypothesis, y_target)
    # cost = cost_func(hypothesis, y_target) # cost func을 미리 지정하는 방법
    cost.backward()
    optimizer.step()

    if epoch%10 == 0:
        print('epoch:{}, cost:{:.3f}'.format(epoch, cost.item()))

for p in model.parameters():
    print(p)