"""Test Windows setup"""
import sys
from pathlib import Path

print("🧠 Seizure Sentinel - Windows Setup Test")
print("="*50)

# Test 1: Directories
print("\n1. Checking directories...")
required = ['data/raw', 'data/processed', 'src', 'models', 'results']
for d in required:
    status = "✓" if Path(d).exists() else "✗"
    print(f"   {status} {d}")

# Test 2: Python imports
print("\n2. Testing Python packages...")
packages = [
    ('torch', 'PyTorch'),
    ('numpy', 'NumPy'),
    ('pandas', 'Pandas'),
    ('scipy', 'SciPy'),
]

for pkg, name in packages:
    try:
        __import__(pkg)
        print(f"   ✓ {name}")
    except ImportError:
        print(f"   ✗ {name} - run: pip install {pkg}")

# Test 3: CUDA
try:
    import torch
    cuda_available = torch.cuda.is_available()
    print(f"\n3. GPU Status:")
    print(f"   CUDA available: {cuda_available}")
    if cuda_available:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
except:
    print("\n3. GPU Status: Unknown")

print("\n" + "="*50)
print("Setup test complete!")