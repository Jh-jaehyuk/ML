import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x, diff=False): # default False
    if diff:
        return sigmoid(x) * (1 - sigmoid(x)) #sigmoid diff
    else:
        return 1 / (1 + np.exp(-x)) #sigmoid

# Sigmoid
x = np.arange(-10.0, 10.0, 0.1)
y = sigmoid(x)

fig = plt.figure(figsize=(5, 4))
fig.set_facecolor('white')

plt.plot(x, y)
plt.xlim(-10, 10)
plt.ylim(-0.1, 1.1)
plt.title('Sigmoid', fontsize=20)
plt.xlabel('x', fontsize=10)
plt.ylabel('y', rotation=0, fontsize=10)
plt.yticks([0.0, 0.5, 1.0])
plt.axvline(0.0, color='k')
ax = plt.gca()
ax.yaxis.grid(True)

plt.show()

# Sigmoid diff
y_diff = sigmoid(x, True)

fig = plt.figure(figsize=(5, 4))
fig.set_facecolor('white')

plt.plot(x, y_diff)
plt.xlim(-7, 7)
plt.ylim(-0.1, 0.4)
plt.title('Sigmoid diff', fontsize=20)
plt.xlabel('x', fontsize=10)
plt.ylabel('y', rotation=0, fontsize=10)
plt.yticks([0.0, 0.2, 0.4])
plt.axvline(0.0, color='k')
ax = plt.gca()
ax.yaxis.grid(True)

plt.show()

