"""Complete setup verification"""
import sys
sys.path.append('src')

print("="*60)
print("🧠 SEIZURE SENTINEL - SETUP VERIFICATION")
print("="*60)

# Test 1: Check files exist
print("\n1. Checking code files...")
from pathlib import Path

files = [
    'src/data_loader.py',
    'src/preprocessor.py', 
    'src/model.py',
    'src/train.py',
    'src/evaluate.py',
    'main.py',
    'config.json'
]

all_exist = True
for f in files:
    exists = Path(f).exists()
    status = "✓" if exists else "✗"
    print(f"   {status} {f}")
    if not exists:
        all_exist = False

# Test 2: Test imports
print("\n2. Testing imports...")
try:
    from data_loader import CHBMITLoader
    print("   ✓ data_loader")
except Exception as e:
    print(f"   ✗ data_loader: {e}")
    all_exist = False

try:
    from preprocessor import EEGPreprocessor
    print("   ✓ preprocessor")
except Exception as e:
    print(f"   ✗ preprocessor: {e}")
    all_exist = False

try:
    from model import SeizureDetectionLSTM
    print("   ✓ model")
except Exception as e:
    print(f"   ✗ model: {e}")
    all_exist = False

try:
    from train import SeizureDetectionTrainer
    print("   ✓ train")
except Exception as e:
    print(f"   ✗ train: {e}")
    all_exist = False

try:
    from evaluate import SeizureDetectionEvaluator
    print("   ✓ evaluate")
except Exception as e:
    print(f"   ✗ evaluate: {e}")
    all_exist = False

# Test 3: Check dataset
print("\n3. Checking dataset...")
try:
    loader = CHBMITLoader('data/raw/chb-mit')
    patients = loader.get_patient_ids()
    print(f"   ✓ Found {len(patients)} patients: {patients}")
    
    if len(patients) > 0:
        stats = loader.get_dataset_statistics()
        print(f"   ✓ Total seizures: {stats['n_seizures'].sum()}")
        print(f"   ✓ Total hours: {stats['total_duration_hours'].sum():.1f}")
    else:
        print("   ✗ No patients found in data/raw/chb-mit/")
        all_exist = False
except Exception as e:
    print(f"   ✗ Dataset check failed: {e}")
    all_exist = False

# Test 4: Check PyTorch
print("\n4. Testing PyTorch...")
try:
    import torch
    print(f"   ✓ PyTorch {torch.__version__}")
    print(f"   ✓ CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"   ✗ PyTorch: {e}")
    all_exist = False

# Final result
print("\n" + "="*60)
if all_exist:
    print("✅ SETUP COMPLETE - READY TO TRAIN!")
    print("="*60)
    print("\nNext step: Run training")
    print("   python main.py --data_dir data/raw/chb-mit --mode train")
else:
    print("⚠️  SETUP INCOMPLETE - Check errors above")
    print("="*60)