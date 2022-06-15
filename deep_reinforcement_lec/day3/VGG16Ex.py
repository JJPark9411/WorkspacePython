import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms

batch_size = 20
learning_rate = 0.0002
num_epoch = 20

img_dir = 'images'

# ImageFolder를 사용하면 img_dir 안에 있는 각 폴더에 따라 이미지를 0, 1, ...로 labeling 해줌
img_data = dset.ImageFolder(img_dir,
                            transforms.Compose([
                                transforms.Resize(226), # 읽어온 이미지를 모두 226x226으로 변환
                                transforms.RandomSizedCrop(224), # 랜덤한 위치에서 이미지를 224x224로 crop
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(degrees=20), # 최대 각도 20도로 무작위 회전
                                transforms.RandomGrayscale(p=0.1), # 0.1의 확률로 grayscale로 가져옴
                                transforms.ToTensor()
                            ]))
train_loader = data.DataLoader(img_data, batch_size=batch_size, shuffle=True)

def conv_2_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
    )
    return model

def conv_3_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
    )
    return model


class VGG16(nn.Module):
    def __init__(self, base_dim, num_class=2):
        super().__init__()
        self.feature = nn.Sequential( # (batch=16, channel=3, height=224, width=224)인 데이터를 입력으로 받음
            conv_2_block(3, base_dim), # 컬러 이미지이므로 input dim이 3 # base_dim 개의 112x112 layer를 출력
            conv_2_block(base_dim, 2*base_dim), # 2*base_dim 개의 56x56 layer를 출력
            conv_3_block(2*base_dim, 4*base_dim), # 4*base_dim 개의 28x28 layer를 출력
            conv_3_block(4*base_dim, 8*base_dim), # 8*base_dim 개의 14x14 layer를 출력
            conv_3_block(8*base_dim, 8*base_dim) # 8*base_dim 개의 7x7 layer를 출력
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(8*base_dim * 7 * 7, 100),
            nn.ReLU(),
            nn.Dropout(), # overfitting 되는 것을 방지
            nn.Linear(100, 20),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(20, num_class)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        y = self.fc_layer(x)
        return y


model = VGG16(base_dim=16)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epoch):
    for x_train, y_train in train_loader:
        optimizer.zero_grad()
        hypothesis = model(x_train)
        loss = loss_func(hypothesis, y_train)
        loss.backward()
        optimizer.step()

    print('epoch:{} loss:{:.4f}'.format(epoch, loss.item()))