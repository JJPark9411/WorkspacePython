import torch
import torch.nn as nn

input = torch.Tensor(1, 1, 28, 28) # (batch, channel, height, width): 28x28 1ch 데이터를 1개 생성

# 1ch 이미지를 input으로 받아서
# 3x3 kernel 32개를 적용 -> conv. layer 32개 출력
conv1 = nn.Conv2d(1, 32, 3, padding=1) # (input channel, kernel, kernel size, padding)

conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # kernel_size를 명시해도, 안 해도 됨

pool = nn.MaxPool2d(2) # 2x2 pooling. stride는 자동으로 (2, 2)로 지정됨

output = conv1(input)
print(output.size()) # (1, 32, 28, 28)

output = conv2(output)
print(output.size()) # (1, 64, 28, 28)

output = pool(output)
print(output.size()) # (1, 64, 14, 14)

output = output.view(output.size(0), -1) # batch(0번째)만 그대로 두고 나머지 차원의 요소를 직렬화
print(output.size()) # (1, 12544) 64*14*14=12544

fc = nn.Linear(12544, 10)
output = fc(output)
print(output.size())