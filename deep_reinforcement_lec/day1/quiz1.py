import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

x_train = torch.FloatTensor( [[73,80,75,65],
                              [93,88,93,88],
                              [89,91,90,76],
                              [96,98,100,99],
                              [73,65,70,100],
                              [84, 98, 90, 100]])

y_train = torch.FloatTensor([[152],[185],[180],[196],[142],[188]])

dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

class MLRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4,1)

    def forward(self, x):
        return self.linear(x)

model = MLRegressionModel()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

for epoch in range(2000):
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples
        hypothesis = model(x_train)
        cost = F.mse_loss(hypothesis, y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print('epoch:{} cost:{:.4f}'.format(epoch, cost.item()))
