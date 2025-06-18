import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"Is CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version (Compiled with PyTorch): {torch.version.cuda}")
    print(f"Current CUDA Device: {torch.cuda.get_device_name(0)}")