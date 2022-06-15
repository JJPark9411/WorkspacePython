import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

batch_size = 32
learning_rate = 0.0002
total_epoch = 20

mnist_train = dset.MNIST(root='MNIST_data/',
                         train=True,
                         transform=transforms.ToTensor(),
                         download=True)

train_loadder = torch.utils.data.DataLoader(mnist_train,
                                            batch_size=batch_size,
                                            shuffle=True)

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

autoencoder = AutoEncoder()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.005)
loss_func = nn.MSELoss()


def add_noise(img):
    #noise = torch.randn(img.size()) * 0.8
    noise = nn.init.normal_(torch.FloatTensor(img.size()), mean=255, std=0.8)
    noisy_img = img + noise
    return noisy_img

def train(autoencoder, train_loader):
    avg_loss = 0
    for i, (x, _) in enumerate(train_loader):
        x_data = add_noise(x).view(-1, 784)
        y = x.view(-1, 784)
        encoded, decoded = autoencoder(x_data)

        loss = loss_func(decoded, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
    return avg_loss / len(train_loader)

for epoch in range(total_epoch):
    loss = train(autoencoder, train_loadder)
    print('epoch:{}  loss:{:.4f}'.format(epoch, loss))


testset = dset.MNIST(root='MNIST_data',
                     train=False,
                     download=True,
                     transform=transforms.ToTensor())

sample_data = testset.data[0].view(-1, 784)
sample_data = sample_data.type(torch.FloatTensor)

original_x = sample_data[0]
noisy_x = add_noise(original_x)
_, recovered_x = autoencoder(noisy_x)

f, a = plt.subplots(1, 3, figsize=(15, 15))
import numpy as np
original_img = np.reshape(original_x.data.numpy(), (28,28))
noisy_img = np.reshape(noisy_x.data.numpy(), (28,28))
recovered_img = np.reshape(recovered_x.data.numpy(), (28,28))

a[0].set_title('original')
a[0].imshow(original_img, cmap='gray')

a[1].set_title('noisy')
a[1].imshow(noisy_img, cmap='gray')

a[2].set_title('recovered')
a[2].imshow(recovered_img, cmap='gray')
plt.show()

