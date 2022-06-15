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
                         download=True) # 28x28 mnist 이미지 데이터

train_loader = torch.utils.data.DataLoader(mnist_train,
                                           batch_size=batch_size,
                                           shuffle=True)


class AutoEncoderNet(nn.Module): # 784(=28x28) -> 20 -> 784(=28x28)
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(784, 20)
        self.decoder = nn.Linear(20, 784)

    def forward(self, x):
        x = x.view(batch_size, -1) # 입력 이미지를 직렬화
        eoutput = self.encoder(x)
        output = self.decoder(eoutput).view(batch_size, 1, 28, 28) # 출력 데이터를 이미지로 변환 (channel=1, 28, 28)
        return output


model = AutoEncoderNet()
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(total_epoch):
    for x_data, y_data in train_loader:
        optimizer.zero_grad()
        hypothesis = model(x_data)
        loss = loss_func(hypothesis, x_data) # autoencoder에서는 input인 x_data가 target이 됨
        loss.backward()
        optimizer.step()

    print('epoch: {} loss: {:.4f}'.format(epoch, loss.item()))

out_img = torch.squeeze(hypothesis.data)
for i in range(3):
    plt.imshow(torch.squeeze(x_data[i]).numpy(), cmap='gray')
    plt.figure()
    plt.imshow(out_img[i].numpy(), cmap='gray')
    plt.show()