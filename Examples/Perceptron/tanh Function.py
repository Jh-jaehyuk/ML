import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x, diff=False): # default False
    if diff:
        return sigmoid(x) * (1 - sigmoid(x)) #sigmoid diff
    else:
        return 1 / (1 + np.exp(-x)) #sigmoid

def tanh(x, diff=False):
    if diff:
        return (1 + tanh(x)) * (1 - tanh(x))
    else:
        return np.tanh(x)

# tanh
x = np.arange(-5.0, 5.0, 0.1)
y = tanh(x)

fig = plt.figure(figsize=(5, 4))
fig.set_facecolor('white')

plt.plot(x, y)
plt.xlim(-5, 5)
plt.ylim(-1.1, 1.1)
plt.title('tanh', fontsize=20)
plt.xlabel('x', fontsize=10)
plt.ylabel('y', rotation=0, fontsize=10)
plt.yticks([-1.0, 0.0, 1.0])
plt.axvline(0.0, color='k')
ax = plt.gca()
ax.yaxis.grid(True)

plt.show()

# tanh diff
y_diff = tanh(x, True)

fig = plt.figure(figsize=(5, 4))
fig.set_facecolor('white')

plt.plot(x, y)
plt.xlim(-5, 5)
plt.ylim(-1.1, 1.1)
plt.title('tanh diff', fontsize=20)
plt.xlabel('x', fontsize=10)
plt.ylabel('y', rotation=0, fontsize=10)
plt.yticks([-1.0, 0.0, 1.0])
plt.axvline(0.0, color='k')
ax = plt.gca()
ax.yaxis.grid(True)

plt.show()

# sig diff & tanh diff
x = np.arange(-10.0, 10.0, 0.1)
y_sig_diff = sigmoid(x, True)
y_tanh_diff = tanh(x, True)

fig = plt.figure(figsize=(5, 4))
plt.plot(x, y_sig_diff, c='blue', linestyle='--', label='sigmoid diff')
plt.plot(x, y_tanh_diff, c='red', label='tanh diff')
plt.xlim(-6.5, 6.5)
plt.ylim(-0.5, 1.2)
plt.title('Sigmoid diff & tanh diff', fontsize=20)
plt.xlabel('x', fontsize=10)
plt.ylabel('y', rotation=0, fontsize=10)
plt.legend(loc='upper right')
plt.axvline(0.0, color='k')
ax = plt.gca()
ax.yaxis.grid(True)
ax.xaxis.grid(True)

plt.show()