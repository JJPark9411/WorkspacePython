import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets # mnist datasets이 있는 라이브러리
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

torch.manual_seed(777)

# MNIST 데이터: 28x28짜리 이미지와 0~9 사이의 숫자 쌍 (input: 784, output: 10)
mnist_train = dsets.MNIST(root='MNIST_data',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True) # train 60,000개, test 10,000개 중 train 데이터 로드
mnist_test = dsets.MNIST(root='MNIST_data',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True) # test 데이터 로드

data_loader = DataLoader(dataset=mnist_train,
                         batch_size=20,
                         shuffle=True,
                         drop_last=True)

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(15):
    avg_loss = 0
    for x_train, y_train in data_loader:
        x_train = x_train.view(-1, 28*28) # 28x28 이미지를 1x784로 직렬화

        optimizer.zero_grad()
        hypothesis = model(x_train)
        loss = loss_func(hypothesis, y_train)
        loss.backward()
        optimizer.step()
        avg_loss += loss/len(data_loader)
    print('epoch: {} avg_loss: {:.4f}'.format(epoch, avg_loss))


import matplotlib.pyplot as plt
import random

with torch.no_grad():
    x_test = mnist_test.test_data.view(-1, 28*28).float() # 이미지 데이터를 직렬화 후 float으로 변환
    y_test = mnist_test.test_labels

    prediction = model(x_test)
    correction = torch.argmax(prediction, dim=1) == y_test
    accuracy = correction.float().mean()
    print('accuracy: ', accuracy.item())
    print()

    r = random.randint(0, len(mnist_test)-1)
    x_single_data = mnist_test.test_data[r:r+1].view(-1, 28*28).float()
    y_single_data = mnist_test.test_labels[r:r+1]

    print('label: ', y_single_data.item())
    s_prediction = model(x_single_data)
    print('prediction: ', torch.argmax(s_prediction, dim=1).item())

    plt.imshow(mnist_test.test_data[r:r+1].view(28, 28), cmap='gray')
    plt.show()