from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

dataset = fetch_california_housing()
x_data = dataset.data
x_data = np.asarray(x_data, dtype=np.float32)
y_data = dataset.target
y_data = np.asarray(y_data, dtype=np.float32)

plt.figure(1)
plt.plot(x_data, y_data, 'ro')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Dataset')
plt.show()

# Hyper-parameters define
input_size = 1
output_size = 1
num_epochs = 100
learning_rate = 0.1

# Linear Regression model
model = nn.Linear(input_size, output_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

print(x_data.shape) #(20640, 8)
print(y_data.shape) #(20640,)

# train model
for epoch in range(1, num_epochs + 1):
    # Convert arrays to torch tensors
    inputs = torch.from_numpy(x_data)
    targets = torch.from_numpy(y_data)

    # Predict outputs with the linear model.
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Compute gradients and update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}.')