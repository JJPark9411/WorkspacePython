import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

torch.manual_seed(777)

mnist_train = dsets.MNIST(root='MNIST_data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

mnist_test = dsets.MNIST(root='MNIST_data',
                            train=False,
                            transform=transforms.ToTensor(),
                            download=True)

data_loader = DataLoader(dataset=mnist_train,
                            batch_size=20,
                            shuffle=True,
                            drop_last=True)

class CNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Linear(64 * 7 * 7, 10)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return y


model = CNNet()
loss_func = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    avg_loss = 0
    for x_train, y_train in data_loader:
        optimizer.zero_grad()
        hypothesis = model(x_train)
        loss = loss_func(hypothesis, y_train)
        loss.backward()
        optimizer.step()

        avg_loss += loss / len(data_loader)

    print(f'epoch:{epoch}, avg_loss:{avg_loss:.4f}')

import matplotlib.pyplot as plt
import random

with torch.no_grad():
    x_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float() # 직렬화되지 않은 데이터를 사용
    y_test = mnist_test.test_labels

    prediction = model(x_test)
    correction = torch.argmax(prediction, dim=1) == y_test
    accuracy = correction.float().mean()
    print('accuracy:', accuracy.item())
    print()


# r = random.randint(0, len(mnist_test)-1)
# x_sigle_data = mnist_test.test_data[r:r+1].view(1, 1, 28 , 28).float()
# y_sigle_data = mnist_test.test_labels[r:r+1]
#
# print('label:',y_sigle_data.item())
# s_prediction = model(x_sigle_data)
# print('prediction:',torch.argmax(s_prediction, dim=1).item())
#
# plt.imshow(mnist_test.test_data[r:r+1].view(28,28), cmap='gray')
# plt.show()