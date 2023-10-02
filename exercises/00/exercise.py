import torch

# 2. Create a random tensor with shape `(7, 7)`.
print("2.")
x = torch.rand([7, 7])
print(x)

# 3. Perform a matrix multiplication on the tensor from 2 with
# another random tensor with shape `(1, 7)` (hint: you may have to transpose the second tensor).
print("3.")

y = torch.rand([1, 7])
z = x @ y.T
print(z)

z = x.matmul(y.T)
print(z)

# 4. Set the random seed to `0` and do exercises 2 & 3 over again.
print("4.")

RANDOM_SEED = 0
torch.manual_seed(seed=RANDOM_SEED)
x = torch.rand([7, 7])
torch.manual_seed(seed=RANDOM_SEED)
y = torch.rand([1, 7])

print(x @ y.T)
# 5. Speaking of random seeds, we saw how to set it with `torch.manual_seed()`
# but is there a GPU equivalent? (hint: you'll need to look into the
# documentation for `torch.cuda` for this one). If there is, set the GPU random seed to `1234`.
print("5.")

torch.cuda.manual_seed(seed=1234)
print("Set seed...")

# 6. Create two random tensors of shape `(2, 3)` and send them both to the GPU
# (you'll need access to a GPU for this). Set `torch.manual_seed(1234)`
# when creating the tensors (this doesn't have to be the GPU random seed).
print("6.")

# Doing this since my macbook does not have a gpu available
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(seed=1234)

x = torch.rand([2, 3])

torch.manual_seed(seed=1234)

y = torch.rand([2, 3])

x_gpu = x.to(device)
y_gpu = y.to(device)
print(x_gpu)
print(y_gpu)

# 7. Perform a matrix multiplication on the tensors you created in 6
# (again, you may have to adjust the shapes of one of the tensors).
print("7.")

z = x @ y.T
print(z)

# 8. Find the maximum and minimum values of the output of 7.
print("8.")
print(z.max())
print(z.min())

# 9. Find the maximum and minimum index values of the output of 7.
print("9.")
print(z.argmax())
print(z.argmin())

# 10. Make a random tensor with shape `(1, 1, 1, 10)` and then create
# a new tensor with all the `1` dimensions removed to be left with a
# tensor of shape `(10)`. Set the seed to `7` when you create it and
# print out the first tensor and it's shape as well as the second tensor
# and it's shape.
print("10.")

torch.manual_seed(seed=7)
x = torch.rand([1, 1, 1, 10])

print(x.shape)
x = x.squeeze()
print(x.shape)
