import torch
from torch import nn
import matplotlib.pyplot as plt

# THIS MODULE REPLICATES COMMON NONLINEAR ACTIVATION FUNCTIONS

A = torch.arange(-10, 10, 1, dtype=torch.float32)
# plt.plot(A)
# plt.show()

# Replicate ReLU function
def relu(x):
  return torch.maximum(torch.tensor(0), x)

# plt.plot(relu(A))

def sigmoid(x):
  return 1 / (1 + torch.exp(-x))

# plt.plot(sigmoid(A))


plt.show()