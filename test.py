import sys
import torch
import platform
import os

print("OS:", platform.platform())
print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
