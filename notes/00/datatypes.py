import torch

# Default datatype for tensors is float32
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None,  # defaults to None, which is torch.float32 or whatever datatype is passed
                               device=None,  # defaults to None, which uses the default tensor type
                               requires_grad=False)  # if True, operations performed on the tensor are recorded

print(float_32_tensor.shape, float_32_tensor.dtype, float_32_tensor.device)

# Create a float16 tensor
float_16_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=torch.float16)  # torch.half would also work

print(float_16_tensor.dtype)

# Getting Info on Tensors
print()
# Create a tensor
some_tensor = torch.rand(3, 4)

# Find out details about it
print(some_tensor)
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Datatype of tensor: {some_tensor.dtype}")
# will default to CPU
print(f"Device tensor is stored on: {some_tensor.device}")
