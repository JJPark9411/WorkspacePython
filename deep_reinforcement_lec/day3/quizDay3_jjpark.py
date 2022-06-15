import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import numpy as np

batch_size = 64
learning_rate = 0.0002
total_epoch = 10

fashion_mnist_train = dset.FashionMNIST(root='MNIST_data/',
                                        train=True,
                                        transform=transforms.ToTensor(),
                                        download=True) # 28x28 mnist 이미지 데이터

train_loader = torch.utils.data.DataLoader(fashion_mnist_train,
                                           batch_size=batch_size,
                                           shuffle=True)


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # encoder layer
        self.encoder = nn.Sequential(
            nn.Linear(784, 64),
            nn.ReLU(),
            nn.Linear(64, 20)
        )
        # decoder layer
        self.decoder = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 784)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


model = AutoEncoder()
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(total_epoch):
    for x_data, _ in train_loader:
        x_data = x_data.view(-1, 784)
        optimizer.zero_grad()
        encoded, decoded = model(x_data)

        loss = loss_func(decoded, x_data)
        loss.backward()
        optimizer.step()

    print('epoch: {} loss: {:.4f}'.format(epoch, loss.item()))

    plt.figure(epoch)
    for i in range(5):
        plt.subplot(5, 2, 2*i+1)
        plt.title('[{}]original'.format(i))
        original_img = np.reshape(torch.squeeze(x_data[i]).detach().numpy(), (28, 28))
        plt.imshow(original_img, cmap='gray')

        plt.subplot(5, 2, 2*i+2)
        plt.title('[{}]decoded'.format(i))
        decoded_img = np.reshape(torch.squeeze(decoded[i]).detach().numpy(), (28, 28))
        plt.imshow(decoded_img, cmap='gray')

    plt.show()