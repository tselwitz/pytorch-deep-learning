import torch
import random

# # Set the random seed
# try changing this to different values and see what happens to the numbers below
RANDOM_SEED = 42
torch.manual_seed(seed=RANDOM_SEED)
random_tensor_C = torch.rand(3, 4)

# Have to reset the seed every time a new rand() is called
# Without this, tensor_D would be different to tensor_C
# try commenting this line out and seeing what happens
torch.random.manual_seed(seed=RANDOM_SEED)
random_tensor_D = torch.rand(3, 4)

print(f"Tensor C:\n{random_tensor_C}\n")
print(f"Tensor D:\n{random_tensor_D}\n")
print(f"Does Tensor C equal Tensor D? (anywhere)")
random_tensor_C == random_tensor_D
