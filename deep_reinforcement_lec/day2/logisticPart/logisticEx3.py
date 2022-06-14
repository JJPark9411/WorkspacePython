import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)


class LogisticClass(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))


# model = LogisticClass()
model = nn.Sequential(
    nn.Linear(2, 1),
    nn.Sigmoid()
) # Sequential을 이용해 위 클래스와 동일한 모델 생성 가능

optimizer = optim.SGD(model.parameters(), lr=1)
for epoch in range(1000):
    hypothesis = model(x_train)
    loss = F.binary_cross_entropy(hypothesis, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch%10 == 0:
        prediction = hypothesis > 0.5 # boolean 출력
        correct_prediction = prediction.float() == y_train # boolean 값을 float으로 변환 후 y_train과 비교
        accuracy = correct_prediction.sum().item() / len(correct_prediction)

        print(f'epoch:{epoch}, loss:{loss.item():.4f}, accuracy:{accuracy*100:2.2f}')