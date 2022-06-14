import numpy as np

wsum = np.array([0.3, 2.9, 4.1])

def softmax(ws):
    exp_a = np.exp(ws)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

output = softmax(wsum)
print(output)
print(output.sum()) # softmax 연산값의 총합은 1