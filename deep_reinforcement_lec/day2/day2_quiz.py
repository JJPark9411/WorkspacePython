import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split

torch.manual_seed(777)

iris = pd.read_csv('iris.csv')
print(iris.head(10))
print(iris.Name.unique())

mappings = {
    'Iris-setosa':0,
    'Iris-versicolor':1,
    'Iris-virginica':2
}
iris['Name'] = iris['Name'].map(lambda x: mappings[x]) # string으로 되어 있는 Name을 정수로 mapping하여 변환
print()
print(iris.head(20))
print(iris.tail(20))

x = iris.drop('Name', axis=1).values # Name을 제외하고 values로 변환
y = iris['Name'].values # Name을 values로 전환

print(x)
print(y)

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1, random_state=48)

x_train = torch.FloatTensor(train_x)
x_test = torch.FloatTensor(test_x)
y_train = torch.LongTensor(train_y)
y_test = torch.LongTensor(test_y)

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16,12)
        self.fc3 = nn.Linear(12,3)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = self.fc3(x)

        return y

model = NN()
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 100
loss_arr = []

for i in range(epochs):
    hypothesis = model(x_train)
    loss = loss_func(hypothesis, y_train)
    loss_arr.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(i, loss.item())

import matplotlib.pyplot as plt
plt.plot(loss_arr)
plt.show()

print()
preds = []
with torch.no_grad():
    for val in x_test:
        hypothesis = model(val)
        preds.append(hypothesis.argmax().numpy())

df = pd.DataFrame({'y':y_test, 'pred':preds})
df['correct'] = [1 if corr == pred else 0 for corr, pred in zip(df['y'], df['pred'])]
print(df)



















