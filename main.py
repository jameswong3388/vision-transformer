import torch

if torch.backends.mps.is_available():
    mps_device = torch.device("cuda")
    x = torch.ones(1, device=mps_device)
    print(x)
else:
    print("MPS device not found.")
