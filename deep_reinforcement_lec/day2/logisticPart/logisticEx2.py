import torch
import torch.nn.functional as F
import torch.optim as optim

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

w = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

hypothesis = 1 / (1 + torch.exp(-(x_train.matmul(w) + b)))
print(hypothesis)

hypothesis2 = torch.sigmoid(x_train.matmul(w) + b) # torch에 내장된 sigmoid
print(hypothesis2)

losses = -(y_train*torch.log(hypothesis) + (1-y_train)*torch.log(1-hypothesis))
print(losses)

loss = losses.mean()
print(loss)
print()

loss2 = F.binary_cross_entropy(hypothesis, y_train) # torch에 내장된 binary cross entropy (loss function)
print(loss2)

optimizer = optim.SGD([w, b], lr=1)
for epoch in range(1000):
    hypothesis = torch.sigmoid(x_train.matmul(w)+b)
    loss = F.binary_cross_entropy(hypothesis, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch%20 == 0:
        print(f'epoch:{epoch} loss:{loss.item():.4f}')
        print('epoch:{} loss:{:.4f}'.format(epoch, loss.item()))

hypothesis = torch.sigmoid(x_train.matmul(w)+b)
prediction = hypothesis > torch.FloatTensor([0.5])
# prediction = hypothesis > 0.5 # 위와 동일
print(hypothesis)
print()
print(prediction)
