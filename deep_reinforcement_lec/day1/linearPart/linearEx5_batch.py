import torch
import torch.nn as nn
import torch.nn.functional as F

x_train = torch.FloatTensor([[73,80,75],
                            [93,88,93],
                            [89,91,90],
                            [96,98,100],
                            [73,65,70]]) # size 3짜리 데이터 5개
y_train = torch.FloatTensor([[152],[185],[180],[196],[142]]) # 데이터 5개

from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(x_train, y_train) # x와 y 데이터를 합쳐주는 역할
dataloader = DataLoader(dataset, batch_size=2) # batch_size만큼 데이터를 모아 사용

for data in dataloader:
    print(data, end='\n\n') # 2개씩 묶인 data batch가 출력됨

model = nn.Linear(3, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

for epoch in range(20):
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples
        hypothesis = model(x_train)
        cost = F.mse_loss(hypothesis, y_train)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('epoch: {}/{} batch: {}/{} cost: {:.3f}'.format(
            epoch+1, 20, batch_idx+1, len(dataloader), cost.item()
        ))