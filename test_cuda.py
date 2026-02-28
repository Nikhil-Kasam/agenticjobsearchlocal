import torch
print(f"PyTorch:          {torch.__version__}")
print(f"CUDA available:   {torch.cuda.is_available()}")
print(f"GPU count:        {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name:         {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"CUDA capability:  sm_{cap[0]}{cap[1]}")
    t = torch.tensor([1.0, 2.0, 3.0]).cuda()
    print(f"GPU tensor:       {t}")
    print(f"✅ RTX 5090 CUDA acceleration is WORKING!")
else:
    print("❌ CUDA not available — still using CPU")
