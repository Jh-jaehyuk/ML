import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    array_x = x - np.max(x) # 오버플로우 방지를 위해 input - input의 최대값
    exp_x = np.exp(array_x)
    result = exp_x / np.sum(exp_x)
    return result

x = np.arange(-5, 5, 0.1)
y = softmax(x)

fig = plt.figure(figsize=(5, 4))
fig.set_facecolor('white')

plt.plot(x, y)
plt.ylim(0, 0.1)
plt.title('Softmax', fontsize=20)
plt.xlabel('x', fontsize=10)
plt.ylabel('y', rotation=0, fontsize=10)

plt.show()