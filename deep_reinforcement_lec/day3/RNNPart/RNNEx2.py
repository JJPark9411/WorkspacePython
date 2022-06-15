import torch
import torch.nn as nn
import numpy as np

string ='hello pytorch. how long can a rnn cell remember? show me your limit!'
chars = 'abcdefghijklmnopqrstuvwxyz ?!.,:;01' # 위 문장을 one-hot encoding하기 위한 문자 모음. 0은 start, 1은 end의 의미로 사용

char_list = [i for i in chars]
n_letter = len(char_list) # input_size, hidden_size가 되는 크기
print(char_list)
print(n_letter)

n_hidden = 35 # n_letter 값으로 설정
learning_rate = 0.01
total_epoch = 1000


def stringToOneHot(string):
    start = np.zeros(n_letter, dtype=int)
    end = np.zeros(n_letter, dtype=int)
    start[-2] = 1 # [0, 0, 0, ..., 0, 1, 0]
    end[-1] = 1   # [0, 0, 0, ..., 0, 0, 1]

    for i in string:
        idx = char_list.index(i) # char_list에서 i가 있는 index를 가져옴
        odata = np.zeros(n_letter, dtype=int)
        odata[idx] = 1
        start = np.vstack([start, odata])
    output = np.vstack([start, end])
    return output

# print(stringToOneHot('test'))

def oneHotToChar(onehot_d):
    onehot = torch.Tensor.numpy(onehot_d)
    return char_list[onehot.argmax()]

# print(oneHotToChar(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])))


class RNNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size): # output_size가 hidden_size와 다른 경우가 있지만 여기서는 동일하게 구성
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2o = nn.Linear(input_size+hidden_size, output_size)
        self.i2h = nn.Linear(input_size+hidden_size, hidden_size)
        self.act_fn = nn.Tanh()

    def init_hidden(self): # 초기 hidden 값을 초기화하는 함수
        return torch.zeros(1, self.hidden_size)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.act_fn(self.i2h(combined))
        output = self.i2o(combined)
        return output, hidden


rnn = RNNet(n_letter, n_hidden, n_letter)
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

one_hot = torch.from_numpy(stringToOneHot(string)).type_as(torch.FloatTensor())

for epoch in range(total_epoch):
    optimizer.zero_grad()
    hidden = rnn.init_hidden()
    total_loss = 0

    for j in range(one_hot.size()[0]-1):
        input = one_hot[j:j+1, :]
        target = one_hot[j+1]
        output, hidden = rnn(input, hidden)
        loss = loss_func(output.view(-1), target.view(-1)) # .view(-1)로 모두 직렬화
        total_loss += loss

    total_loss.backward()
    optimizer.step()
    if epoch%10 == 0:
        # print('epoch: {} total_loss: {:.4f}'.format(epoch, total_loss.item()))

        start = torch.zeros(1, n_letter)
        start[:, -2] = 1
        with torch.no_grad():
            hidden = rnn.init_hidden()
            input = start
            output_string = ''
            for i in range(len(string)):
                output, hidden = rnn(input, hidden)
                output_string += oneHotToChar(output)
                input = output

        print()
        print(output_string)