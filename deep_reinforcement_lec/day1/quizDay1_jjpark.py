import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader


class CustomLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.w = nn.Parameter(torch.FloatTensor(torch.randn(input_size, output_size)), requires_grad=True)
        self.b = nn.Parameter(torch.FloatTensor(torch.randn(output_size)), requires_grad=True)

    def forward(self, x):
        y = torch.mm(x, self.w) + self.b
        return y


x_train = torch.FloatTensor([[73, 80, 75, 65],
                             [93, 88, 93, 88],
                             [89, 91, 90, 76],
                             [96, 98, 100, 99],
                             [73, 65, 70, 100],
                             [84, 98, 90, 100]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142], [188]])

dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=3)

# for data in dataloader:
#     print(data)

model = CustomLinear(4, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

for epoch in range(20):
    for batch_idx, samples  in enumerate(dataloader):
        x_train, y_train = samples
        hypothesis = model(x_train)
        cost = F.mse_loss(hypothesis, y_train)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('epoch: {}/{}, batch: {}/{}, cost: {:.3f}'.format(
            epoch+1, 20, batch_idx+1, len(dataloader), cost.item()
        ))