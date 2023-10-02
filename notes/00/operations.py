import torch

# Create a tensor of values and add a number to it

T = torch.tensor([1, 2, 3])
print(T + 10)

# Mult 10
print(T * 10)

# Sub and reassign
T -= 10
print(T)

# Add and reassign
T += 10
print(T)

# Builtins
print(torch.multiply(T, 10))
print(torch.add(T, 10))

# Note that the tensor doesn't change because there's no reassignment going on
print(T)

# Matrix multiplication

# Element wise multiplication uses *
T = T * T

# Matrix multiplication is done like this:

T = torch.tensor([1, 2, 3])  # 1x3
R = torch.tensor([[3], [4], [5]])  # 3x1
print(torch.matmul(T, R))  # 1x1 Result


# Shape Errors

# Shapes need to be in the right way
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]], dtype=torch.float32)

tensor_B = torch.tensor([[7, 10],
                         [8, 11],
                         [9, 12]], dtype=torch.float32)

# torch.matmul(tensor_A, tensor_B)  # (this will error)

print(torch.matmul(tensor_A, tensor_B.T))  # Try transposing B

# The operation works when tensor_B is transposed
print(
    f"Original shapes: tensor_A = {tensor_A.shape}, tensor_B = {tensor_B.shape}\n")
print(
    f"New shapes: tensor_A = {tensor_A.shape} (same as above), tensor_B.T = {tensor_B.T.shape}\n")
print(
    f"Multiplying: {tensor_A.shape} * {tensor_B.T.shape} <- inner dimensions match\n")
print("Output:\n")
output = torch.matmul(tensor_A, tensor_B.T)
print(output)
print(f"\nOutput shape: {output.shape}")

print("\n__Squeezing and Reshaping__\n")
# reshaping, squeezing
x = torch.arange(1., 8.)
print(x, x.shape)

print("Reshaped...")
x_reshaped = x.reshape(1, 7)
print(x_reshaped, x_reshaped.shape)

# you can also use "view"
z = x.view(1, 7)
print(z, z.shape)

# Changing a view changes the tensor
z[:, 0] = 5
print(z, x, "<= Note that these are the same")

# Stack tensors on top of each other
# try changing dim to dim=1 and see what happens
x_stacked = torch.stack([x, x, x, x], dim=0)
print(x_stacked)

# squeeze

print(f"Previous tensor: {x_reshaped}")
print(f"Previous shape: {x_reshaped.shape}")

# Remove extra dimension from x_reshaped
x_squeezed = x_reshaped.squeeze()
print(f"\nNew tensor: {x_squeezed}")
print(f"New shape: {x_squeezed.shape}")

print(f"Previous tensor: {x_squeezed}")
print(f"Previous shape: {x_squeezed.shape}")

# unsqueeze

# Add an extra dimension with unsqueeze
x_unsqueezed = x_squeezed.unsqueeze(dim=0)
print(f"\nNew tensor: {x_unsqueezed}")
print(f"New shape: {x_unsqueezed.shape}")

# swap dimensions

# Create tensor with specific shape
x_original = torch.rand(size=(224, 224, 3))

# Permute the original tensor to rearrange the axis order
x_permuted = x_original.permute(2, 0, 1)  # shifts axis 0->1, 1->2, 2->0

print(f"Previous shape: {x_original.shape}")
print(f"New shape: {x_permuted.shape}")
