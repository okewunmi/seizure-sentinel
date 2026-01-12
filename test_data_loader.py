"""Test the data loader"""
import sys
sys.path.append('src')  # Add src to path

from data_loader import CHBMITLoader

print("🧠 Testing Data Loader")
print("="*50)

# Initialize loader
loader = CHBMITLoader('data/raw/chb-mit')

# Get patient IDs
patients = loader.get_patient_ids()
print(f"\nFound {len(patients)} patients: {patients}")

if patients:
    # Get statistics
    stats = loader.get_dataset_statistics()
    print("\nDataset Statistics:")
    print(stats)
    
    print(f"\nTotal seizures: {stats['n_seizures'].sum()}")
    print(f"Total hours: {stats['total_duration_hours'].sum():.1f}")
else:
    print("\n⚠️  No patients found!")
    print("Dataset directory is empty or doesn't exist.")
    print("\nNext steps:")
    print("1. Download CHB-MIT dataset")
    print("2. Place in: data/raw/chb-mit/")
    print("3. Run this script again")