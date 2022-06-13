import torch
import torch.optim as optim

x1_train = torch.FloatTensor([[73],[93],[88],[96],[73]])
x2_train = torch.FloatTensor([[80],[88],[92],[98],[67]])
x3_train = torch.FloatTensor([[75],[94],[90],[100],[70]])
y_train = torch.FloatTensor([[152],[185],[180],[196],[142]])

w1 = torch.zeros((1,1), requires_grad=True)
w2 = torch.zeros((1,1), requires_grad=True)
w3 = torch.zeros((1,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([w1, w2, w3, b], lr=1e-5) # 사용할 weight를 지정

for epoch in range(1000):
    hypothesis = torch.mm(x1_train, w1) + torch.mm(x2_train, w2) + torch.mm(x3_train, w3) + b
    cost = torch.mean((hypothesis-y_train)**2)
    optimizer.zero_grad() # optimizer 초기화를 해야 gradient가 누적되지 않음
    cost.backward() # w1, w2, w3, b에 대한 편미분을 수행
    optimizer.step() # optimizer에서 지정된 weight로 편미분한 값에 lr을 곱해서 w1, w2, w3, b를 업데이트

    if epoch%10 == 0:
        print('epoch:{}, w1:{:.3f}, w2:{:.3f}, w3:{:.3f}, cost:{:.3f}'.format(
            epoch, w1.item(), w2.item(), w3.item(), b.item()
        ))