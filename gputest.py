import torch

# This should return True
print(f"CUDA Available: {torch.cuda.is_available()}")

# This should print 'NVIDIA GeForce RTX 3050 Ti Laptop GPU'
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("System is running on CPU.")