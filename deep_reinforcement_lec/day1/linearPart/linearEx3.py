import torch
import torch.nn as nn

class CustomLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.w = nn.Parameter(torch.FloatTensor(torch.randn(input_size, output_size)), requires_grad=True)
        self.b = nn.Parameter(torch.FloatTensor(torch.randn(output_size)), requires_grad=True)

    def forward(self, x):
        y = torch.mm(x, self.w) + self.b
        return y

x = torch.FloatTensor(torch.randn(16, 10)) # 16x10, 크기 10짜리 벡터 데이터가 16개
model = CustomLinear(10, 5) # input: 10 -> output: 5

hypothesis = model.forward(x)
print(hypothesis)
print()

hypothesis2 = model(x) # hypothesis = model.forward(x)와 동일. x가 forward 함수의 인자로 전달됨
print(hypothesis2)
print('----------------------------------------------------------------------')

for p in model.parameters():
    print(p) # w: 10x5, b: 1x5