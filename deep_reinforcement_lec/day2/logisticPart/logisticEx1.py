import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1+np.exp(-x))

x = np.arange(-5, 5, 0.1)

y1 = sigmoid(0.5*x)
y2 = sigmoid(x)
y3 = sigmoid(2*x)

y1 = sigmoid(x-1.5)
y2 = sigmoid(x)
y3 = sigmoid(x+1.5)

plt.plot(x, y1, 'r', label='x-1.5')
plt.plot(x, y2, 'g', label='x')
plt.plot(x, y3, 'b', label='x+1.5')
plt.legend(loc='best')
plt.plot([0,0], [1,0], ':')
plt.show()
