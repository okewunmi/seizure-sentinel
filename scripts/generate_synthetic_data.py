"""Generate synthetic EEG data for testing"""
import numpy as np
from pathlib import Path

print("Generating synthetic EEG data...")

# Create fake patient directory
data_dir = Path('data/raw/chb-mit/chb01')
data_dir.mkdir(parents=True, exist_ok=True)

# Create a summary file
with open(data_dir / 'chb01-summary.txt', 'w') as f:
    f.write("File Name: chb01_01.edf\n")
    f.write("Number of Seizures in File: 1\n")
    f.write("Seizure Start Time: 2996 seconds\n")
    f.write("Seizure End Time: 3036 seconds\n")

print("✓ Created synthetic patient chb01")
print("✓ Created summary file with seizure annotations")
print("\nNote: You'll need real EDF files to train the model")
print("Download from: https://physionet.org/content/chbmit/1.0.0/")