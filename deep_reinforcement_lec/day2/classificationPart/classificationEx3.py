import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import load_wine

torch.manual_seed(777)
wine = load_wine() # sklearn에서 제공하는 wine 데이터 로드
# print(wine.keys())

wine_data = wine.data[0:130] # 데이터 130개만 추출
wine_target = wine.target[0:130]
# print(wine_data.shape)
# print(wine_target) # 130개 데이터에는 정답이 2가지 종류{0, 1}만 존재

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(wine_data, wine_target, test_size=.2, random_state=48)
print(x_train.shape) # 데이터 104개, feature 13개
print(x_test.shape)

x_train = torch.from_numpy(x_train).float()
# x_train = torch.FloatTensor(x_train) # 위와 동일
y_train = torch.from_numpy(y_train).long()

x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test).long()


class NNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(13, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 32)
        self.fc5 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x)) # fc1 레이어 통과 후 ReLu
        x = F.relu(self.fc2(x)) # fc2 레이터 통과 후 ReLu
        x = F.relu(self.fc3(x))  # fc1 레이어 통과 후 ReLu
        x = F.relu(self.fc4(x))  # fc2 레이터 통과 후 ReLu
        y = self.fc5(x) # fc3 레이어 통과 후 weighted sum 상태로 둠. 이후 cross entropy를 적용하기 위함
        return y


model = NNet()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

train = TensorDataset(x_train, y_train)
train_loader = DataLoader(train, batch_size=16, shuffle=True)

for epoch in range(1000):
    total_loss = 0

    for x_train, y_train in train_loader:
        optimizer.zero_grad()
        hypothesis = model(x_train)
        loss = loss_func(hypothesis, y_train)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch%10 == 0:
        print('epoch: {} total_loss: {:.4f}'.format(epoch, total_loss))

print()
prediction = torch.max(model(x_test), dim=1)[1] # max 함수가 values, indices를 출력하므로 [1]로 indices 값을 선택
correct = prediction == y_test
accuracy = correct.float().mean()
print('accuracy: ', accuracy.item())