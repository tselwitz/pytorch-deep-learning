import torch

# Random tensor of size (3,4)
random_tensor = torch.rand(size=(3, 4))
print(random_tensor, random_tensor.dtype)

# Create a random tensor of size (224, 224, 3)
random_image_size_tensor = torch.rand(size=(224, 224, 3))
print(random_image_size_tensor.shape, random_image_size_tensor.ndim)

# Create a tensor with all zeroes
zeros = torch.zeros(size=(3, 4))
print(zeros, zeros.dtype)

# Create a tensor of ones
ones = torch.ones(size=(3, 4))
print(ones, ones.dtype)

# Create a tensor with values from 0 to 10
zero_to_ten = torch.arange(start=0, end=10, step=1)
print(zero_to_ten)

# Create a zeros tensor with the same size as the zero_to_ten
ten_zeros = torch.zeros_like(input=zero_to_ten)
print(ten_zeros)
