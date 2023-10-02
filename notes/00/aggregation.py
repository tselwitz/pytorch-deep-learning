import torch

x = torch.arange(0, 100, 10)
print(x)

print(f"Minimum: {x.min()}")
print(f"Maximum: {x.max()}")
# print(f"Mean: {x.mean()}") # this will error
# won't work without float datatype
print(f"Mean: {x.type(torch.float32).mean()}")
print(f"Sum: {x.sum()}")

# Returns index of max and min values
print(f"Index where max value occurs: {x.argmax()}")
print(f"Index where min value occurs: {x.argmin()}")

print(x.dtype)

tensor_float16 = x.type(torch.float16)
tensor_int8 = x.type(torch.int8)
