import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

dataset = open("Regression/Linear Regression.txt", 'r')
text = dataset.readlines()
dataset.close()

x_data = []
y_data = []

for t in text:
    data = t.split()
    x_data.append(float(data[0]))
    y_data.append(float(data[1]))

x_data.sort()
y_data.sort()

x_data = np.asarray(x_data, dtype=np.float32)
y_data = np.asarray(y_data, dtype=np.float32)

plt.figure(1)
plt.plot(x_data, y_data, 'ro')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Dataset')
#plt.show()

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

print(x_data.shape)
print(y_data.shape)

if len(x_data.shape) == 1 and len(y_data.shape) == 1:
    x_data = np.expand_dims(x_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

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

predicted = model(torch.from_numpy(x_data)).detach().numpy()
plt.plot(x_data, y_data, 'ro', label='Original data')
plt.plot(x_data, predicted, label='Fitted Line')
plt.legend()
plt.show()