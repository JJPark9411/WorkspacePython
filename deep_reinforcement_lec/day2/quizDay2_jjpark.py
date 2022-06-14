import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split

import pandas as pd

torch.manual_seed(777)

iris = pd.read_csv('iris.csv')

mappings = {
    'Iris-setosa':0,
    'Iris-versicolor':1,
    'Iris-virginica':2
}
iris['Name'] = iris['Name'].map(lambda x:mappings[x])

iris_data = iris.iloc[:, :-1].values
iris_target = iris.iloc[:, -1].values
# print(iris.shape)
# print(iris_data)
# print(iris_target)

x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_target, test_size=0.1, random_state=48)

x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

x_test = torch.FloatTensor(x_test)
y_test = torch.LongTensor(y_test)


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        y = self.layer(x)
        return y


model = NN()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
# optimizer = optim.Adam(model.parameters(), lr=0.01)

train = TensorDataset(x_train, y_train)
train_loader = DataLoader(train, batch_size=20, shuffle=True)

for epoch in range(1001):
    avg_loss = 0
    for x_train, y_train in train_loader:
        optimizer.zero_grad()
        hypothesis = model(x_train)
        loss = loss_func(hypothesis, y_train)
        loss.backward()
        optimizer.step()
        avg_loss += loss / len(train_loader)

    if epoch%10 == 0:
        print('epoch: {} avg_loss: {:.4f}'.format(epoch, avg_loss))

print()
prediction = torch.argmax(model(x_test), dim=1)
correct = prediction == y_test
accuracy = correct.float().mean()
print('accuracy: ', accuracy.item())