import torch
import torch.nn as nn

class CustomLinear2(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        y = self.linear(x)
        return y

x = torch.FloatTensor(torch.randn(16, 10))
model = CustomLinear2(10, 5)
hypothesis = model(x)
print(hypothesis)