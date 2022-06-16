import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

total_epoch = 50
batch_size = 100

trainset = datasets.FashionMNIST(root='FashionMNIST/',
                                 train=True,
                                 download=True,
                                 transform=transforms.ToTensor())
train_loader = DataLoader(
    dataset=trainset,
    batch_size=batch_size,
    shuffle=True
)

GModel = nn.Sequential(
    nn.Linear(64, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 784),
    nn.Tanh()
)

DModel = nn.Sequential(
    nn.Linear(784, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 1),
    nn.Sigmoid()
)

loss_func = nn.BCELoss()

d_optimizer = torch.optim.Adam(DModel.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(GModel.parameters(), lr=0.0002)

for epoch in range(total_epoch):
    for images, _ in train_loader:
        images = images.view(batch_size, -1)

        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        outputs = DModel(images)
        d_loss_real = loss_func(outputs, real_labels)

        z = torch.randn(batch_size, 64)
        fake_images = GModel(z)
        outputs = DModel(fake_images)
        d_loss_fake = loss_func(outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        fake_images = GModel(z)
        outputs = DModel(fake_images)
        g_loss = loss_func(outputs, real_labels)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    print('epoch:{}/{} d_loss:{}, g_loss:{}'.format(epoch, total_epoch, d_loss.item(), g_loss.item()))


z = torch.randn(batch_size, 64)
fake_images = GModel(z)

import numpy as np
for i in range(3):
    fake_images_img = np.reshape(fake_images.data.numpy()[i], (28,28))
    plt.imshow(fake_images_img, cmap='gray')
    plt.show()







