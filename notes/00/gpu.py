import torch

print(torch.cuda.is_available())
print(torch.cuda.device_count())

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)
