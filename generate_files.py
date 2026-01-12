"""
Automatic file generator for Seizure Sentinel
Run this to create all Python files with code
"""

import os
from pathlib import Path

# Create directories
dirs = [
    'data/raw', 'data/processed', 'data/annotations',
    'notebooks', 'src', 'models', 'results', 'scripts'
]

print("Creating directories...")
for dir_path in dirs:
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    print(f"  ✓ {dir_path}")

print("\n✓ All directories created!")
print("\nNext: Copy the Python code files from the artifacts into src/")
print("Files needed:")
print("  - src/data_loader.py")
print("  - src/preprocessor.py")
print("  - src/model.py")
print("  - src/train.py")
print("  - src/evaluate.py")
print("  - main.py")