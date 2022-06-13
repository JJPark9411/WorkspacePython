import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset): # Dataset을 상속
    def __init__(self):
        self.x_data = [[73,80,75],
                       [93,88,93],
                       [89,91,90],
                       [96,98,100],
                       [73,65,70]]
        self.y_data = [[152],[185],[180],[196],[142]]

    # CustomDataset을 TensorDataset처럼 사용하기 위해 필요함
    def __len__(self):
        return len(self.x_data)

    # CustomDataset을 TensorDataset처럼 사용하기 위해 필요함
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y

dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

for data in dataloader:
    print(data)

model = nn.Linear(3, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

for epoch in range(20):
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples
        hypothesis = model(x_train)
        model.zero_grad()
        cost = F.mse_loss(hypothesis, y_train)
        cost.backward()
        optimizer.step()

        print('epoch: {}/{}, batch: {}/{}, cost: {:.3f}'.format(
            epoch+1, 20, batch_idx+1, len(dataloader), cost.item()
        ))
        for parameter in model.parameters():
            print(parameter[0])
        print()

