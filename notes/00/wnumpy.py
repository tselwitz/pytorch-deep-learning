import torch
import numpy as np

# NumPy array to tensor
array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array)
print(array, tensor)

# Change the array, keep the tensor
array = array + 1
print(array, tensor)

# Tensor to NumPy array
tensor = torch.ones(7)  # create a tensor of ones with dtype=float32
numpy_tensor = tensor.numpy()  # will be dtype=float32 unless changed
tensor, numpy_tensor

# Change the tensor, keep the array the same
tensor = tensor + 1
tensor, numpy_tensor
