import torch

print(torch.cuda.is_available())
if torch.cuda.device_count() > 0:
    print("Let's use", torch.cuda.device_count(), "GPUs!")

