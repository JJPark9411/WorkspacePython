import torch
import torch.nn as nn
import string
import random

import unidecode

all_characters = string.printable
print(all_characters)
n_characters = len(all_characters)
print(n_characters) # 100

file = unidecode.unidecode(open('input.txt').read()) # unicode 형식의 text 파일 읽기

total_epoch = 2000
chunk_len = 200 # 잘라오는 문자열의 길이
hidden_size = 100
batch_size = 1
num_layer = 2
embedding_size = 70 # embedding vector 크기
learning_rate = 0.002


def random_chunk(): # input.txt에서 무작위 문자열을 추출하는 함수
    start_index = random.randint(0, len(file)-chunk_len)
    end_index = start_index + chunk_len + 1
    return file[start_index:end_index]

print(random_chunk())

def char_tensor(string): # input으로 사용할 문자열을 index로 구성된 tensor로 변환하는 함수
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    return tensor

print(char_tensor('good')) # tensor([16, 24, 24, 13])

class RNNet(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers=1): # num_layers는 RNN 셀로 구성된 layer의 개수
        super().__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.encoder = nn.Embedding(self.input_size, embedding_size)
        self.rnn = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers)
        self.fc = nn.Linear(hidden_size, self.output_size)

    def init_hidden(self): # LSTM은 초기 hidden state, cell state 값이 필요
        hidden = torch.zeros(self.num_layers, batch_size, hidden_size)
        cell = torch.zeros(self.num_layers, batch_size, hidden_size)
        return hidden, cell

    def forward(self, input, hidden, cell):
        x = self.encoder(input.view(1, -1))
        out, (hidden, cell) = self.rnn(x, (hidden, cell))
        out = self.fc(out.view(batch_size, -1))
        return out, hidden, cell


model = RNNet(input_size=n_characters,
              embedding_size=embedding_size,
              hidden_size=hidden_size,
              output_size=n_characters,
              num_layers=num_layer)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_func = nn.CrossEntropyLoss()


def test():
    start_str = 'b'
    input = char_tensor(start_str)
    hidden, cell = model.init_hidden()

    for i in range(200):
        output, hidden, cell = model(input, hidden, cell)
        output_dist = output.data.view(-1).div(0.8).exp() # 0.8로 나눈 후 exponential 적용
        top_i = torch.multinomial(output_dist, 1)[0] # argmax로 output의 최대값을 선택하면 고정된 패턴만 나오므로 확률분포 내에서 선택
        predicted_char = all_characters[top_i]
        print(predicted_char, end='')
        input = char_tensor(predicted_char)

def random_train_set():
    chunk = random_chunk()
    input = char_tensor(chunk[:-1]) # chunk의 처음부터 마지막-1번째까지
    target = char_tensor(chunk[1:])
    return input, target


for epoch in range(total_epoch):
    input, label = random_train_set()
    hidden, cell = model.init_hidden()

    loss = torch.tensor([0]).type(torch.FloatTensor)

    optimizer.zero_grad()
    for j in range(chunk_len-1): # 문자열에서 문자 1개씩 뽑아 학습
        x_train = input[j]
        target = label[j].unsqueeze(0).type(torch.LongTensor)
        output, hidden, cell = model(x_train, hidden, cell)
        loss += loss_func(output, target)
    # loss.backward()
    # optimizer.step()

    if epoch%50 == 0:
        test()
        print('\n', '='*100)

