import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

total_epoch = 50
batch_size = 100

trainset = datasets.FashionMNIST(root='FashionMNIST',
                                 train=True,
                                 download=True,
                                 transform=transforms.ToTensor())

train_loader = DataLoader(dataset=trainset,
                          batch_size=batch_size,
                          shuffle=True)

# Generator
GModel = nn.Sequential(
    nn.Linear(64, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 784),
    nn.Tanh() # (-1, 1) 범위의 출력
)

# Discriminator
# 28x28 이미지를 입력으로 받음
DModel = nn.Sequential(
    nn.Linear(784, 256),
    nn.LeakyReLU(0.2), # 양수 부분 기울기의 0.2에 해당하는 크기로 음수 부분 기울기를 설정. 음수 데이터를 얼마나 활용할 것인지를 의미
    nn.Linear(256, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 1),
    nn.Sigmoid() # 진짜인지 가짜인지에 대한 확률을 출력
)

loss_func = nn.BCELoss() # Binary Cross Entropy 사용

# 각 모델이 따로 학습을 해야 하므로 각각의 optimizer가 있어야 함
d_optimizer = torch.optim.Adam(DModel.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(GModel.parameters(), lr=0.0002)

for epoch in range(total_epoch):
    for images, _ in train_loader:
        images = images.view(batch_size, -1)

        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # discriminator 학습
        outputs = DModel(images)
        d_loss_real = loss_func(outputs, real_labels) # discriminator가 real image를 real로 판별하도록 discriminator를 학습

        z = torch.randn(batch_size, 64)
        fake_images = GModel(z)
        outputs = DModel(fake_images)
        d_loss_fake = loss_func(outputs, fake_labels) # discriminator가 fake image를 fake로 판별하도록 discriminator를 학습

        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # generator 학습
        fake_images = GModel(z)
        outputs = DModel(fake_images)
        g_loss = loss_func(outputs, real_labels) # # discriminator가 fake image를 real로 판별하도록 generator를 학습

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    print('epoch:{}/{} d_loss:{}, g_loss:{}'.format(epoch, total_epoch, d_loss.item(), g_loss.item()))


z = torch.randn(batch_size, 64)
fake_images = GModel(z)

import numpy as np
for i in range(3):
    fake_images_img = np.reshape(fake_images.data.numpy()[i], (28, 28))
    plt.imshow(fake_images_img, cmap='gray')
    plt.show()
