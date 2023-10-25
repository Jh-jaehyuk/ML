import numpy as np
import matplotlib.pyplot as plt

def Leaky_ReLU(x):
    return np.maximum(0.01 * x, x)

x = np.arange(-200.0, 200.0, 0.1)
y = Leaky_ReLU(x)

fig = plt.figure(figsize=(5, 4))
fig.set_facecolor('white')
plt.ylim(-5, 50)
plt.title('Leaky_ReLU', fontsize=20)
plt.xlabel('x', fontsize=10)
plt.ylabel('y', rotation=0, fontsize=10)
plt.axvline(0.0, color='k', linestyle='--', alpha=0.8)
plt.axhline(0.0, color='k', linestyle='--', alpha=0.8)
plt.plot(x, y)

plt.show()