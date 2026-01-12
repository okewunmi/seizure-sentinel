@echo off
echo ========================================
echo Seizure Sentinel - Windows Setup
echo ========================================
echo.

echo Creating directories...
mkdir data\raw 2>nul
mkdir data\processed 2>nul
mkdir data\annotations 2>nul
mkdir notebooks 2>nul
mkdir src 2>nul
mkdir models 2>nul
mkdir results 2>nul
mkdir scripts 2>nul

echo ✓ Directories created!
echo.

echo Creating .gitignore...
(
echo # Data
echo data/raw/
echo data/processed/
echo.
echo # Python
echo __pycache__/
echo *.pyc
echo *.pyo
echo *.egg-info/
echo .ipynb_checkpoints/
echo.
echo # Models
echo models/*.pt
echo models/*.pth
echo.
echo # Environment
echo venv/
echo env/
echo.
echo # IDE
echo .vscode/
echo .idea/
echo.
echo # System
echo .DS_Store
) > .gitignore

echo ✓ .gitignore created!
echo.

echo Creating empty Python files...
type nul > src\__init__.py
echo ✓ src/__init__.py

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo Next steps:
echo 1. Install Python dependencies
echo 2. Copy the Python code files
echo 3. Download CHB-MIT dataset
echo.
pause