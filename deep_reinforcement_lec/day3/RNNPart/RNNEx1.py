import numpy as np

# batch는 1이라 가정
timestep = 10
input_size = 4
hidden_size = 8

inputs = np.random.random((timestep, input_size)) # (batch=1, timestep=10, input_size=4)
hidden_state_t = np.zeros(hidden_size) # 처음에 들어오는 hidden값 초기화. 이후에 업데이트 되며 다음 노드에 전달함

wx = np.random.random((input_size, hidden_size)) # input과 hidden을 이어주는 weights
wh = np.random.random((hidden_size, hidden_size)) # hidden과 hidden을 이어주는 weights
b = np.random.random((hidden_size, ))

total_hidden_state = []

for input_t in inputs:
    output_t = np.tanh(np.dot(input_t, wx) + np.dot(hidden_state_t, wh) + b)
    total_hidden_state.append(list(output_t))
    hidden_state_t = output_t # 노드의 출력이 다음 노드의 hidden 입력으로 들어감

total_hidden_state = np.stack(total_hidden_state, axis=0)
print(total_hidden_state.shape) # 10x8 = (timestep, hidden_size)