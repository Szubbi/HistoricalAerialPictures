import torch

if torch.cuda.is_available():
    print("GPU is available for training.")
    print(f"Using device: {torch.cuda.get_device_name(0)}")
else:
    print("❌ GPU is not available. Training will use CPU.")