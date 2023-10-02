import torch

# Scalar
print("\nScalar:")

# Create a scalar
scalar = torch.tensor(7)
print(scalar)

# Check the dimensions
print("Dimensions:", scalar.ndim)

# Retrieve the value
print("Value:", scalar.item())

# Vector
print("\nVectors:")
vector = torch.tensor([7, 7])
print(vector)

print("Dimensions:", vector.ndim)

print("Shape:", vector.shape)

# Matrix
print("\nMATRIX:")
MATRIX = torch.tensor([[7, 8], [9, 10]])
print(MATRIX)

print("Dimensions:", MATRIX.ndim)

print("Shape:", MATRIX.shape)

# tensor
print("\nTENSOR:")
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])
print(TENSOR)

print("Dimensions:", TENSOR.ndim)

print("Shape:", TENSOR.shape)
