import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(777)

x_train=[[1,2,1,1],
         [2,1,3,2],
         [3,1,3,4],
         [4,1,5,5],
         [1,7,5,5],
         [1,2,5,6],
         [1,6,6,6],
         [1,7,7,7]] # input: 4

y_train = [2,2,2,1,1,1,0,0] # output: 3 {0, 1, 2}

x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x)


model = NN()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(1000):
    hypothesis = model(x_train)
    loss = loss_func(hypothesis, y_train) # nn.CrossEntropyLoss()를 사용하므로 y_train이 정수여야 함
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch%10 == 0:
        print('epoch: {} loss: {:.3f}'.format(epoch, loss))
