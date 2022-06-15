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


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # encoder layer
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), # channel=1, 28x28 이미지 입력 -> channel=16, 28x28 conv layer 출력
            nn.ReLU(),
            nn.BatchNorm2d(16), # Conv2d에서 출력하는 channel 수를 맞춰야 함
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1), # channel=64, 28x28 conv layer 출력
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2) # channel=64, 14x14 conv layer 출력
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2), # channel=128, 7x7 conv layer 출력
            nn.Conv2d(128, 256, 3, padding=1), # channel=256, 7x7 conv layer 출력
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        out = x.view(batch_size, -1) # 데이터 직렬화
        return out

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # decoder layer
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            # channel=256, 7x7 conv layer 입력 -> channel=128, 14x14 conv layer 출력
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1), # channel=64, 14x14 conv layer 출력
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=3, stride=1, padding=1), # channel=16, 14x14 conv layer 출력
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1), # channel=1, 28x28 conv layer 출력
            nn.ReLU()
        )

    def forward(self, x):
        x = x.view(batch_size, 256, 7, 7) # encoder에서 직렬화된 출력을 channel=256, 7x7로 변환
        x = self.layer1(x)
        out = self.layer2(x)
        return out


# layer class가 2개일 때 학습시키는 방법
encoder = Encoder()
decoder = Decoder()

parameters = list(encoder.parameters()) + list(decoder.parameters()) # 두 클래스에서 사용한 layer의 parameter를 list 객체로 하면 됨

loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(parameters, lr=learning_rate)

for epoch in range(total_epoch):
    for j, (x_data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        output = encoder(x_data)
        output = decoder(output)

        loss = loss_func(output, x_data)
        loss.backward()
        optimizer.step()

        if j%100 == 0:
            print('[{}/{}] loss: {:4f}'.format(epoch+1, j+1, loss.item()))

out_img = torch.squeeze(output.data)
for i in range(3):
    plt.subplot(1, 2, 1)
    plt.imshow(torch.squeeze(x_data[i]).numpy(), cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(out_img[i].numpy(), cmap='gray')
    plt.show()