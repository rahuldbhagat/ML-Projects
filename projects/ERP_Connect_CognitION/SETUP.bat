@echo off
echo ============================================================
echo ERP CONNECT COGITION v2.0 - SETUP
echo ============================================================
echo.

echo This will set up your environment for CPU-based training
echo Estimated time: 10-15 minutes
echo.
pause

echo [1/5] Checking Python...
python --version
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.11
    pause
    exit /b 1
)
echo   ✓ Python found
echo.

echo [2/5] Creating virtual environment...
if exist "venv" (
    echo   Virtual environment already exists
) else (
    python -m venv venv
    echo   ✓ Created
)
echo.

echo [3/5] Activating environment...
call venv\Scripts\activate.bat
echo   ✓ Activated
echo.

echo [4/5] Upgrading pip...
python -m pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org
echo   ✓ Pip upgraded
echo.

echo [5/5] Installing packages...
echo   Installing PyTorch CPU...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --trusted-host download.pytorch.org --trusted-host pypi.org --trusted-host files.pythonhosted.org
echo   Installing other packages...
pip install -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org

if errorlevel 1 (
    echo.
    echo ERROR: Package installation failed
    pause
    exit /b 1
)

echo.
echo ============================================================
echo ✅ SETUP COMPLETE!
echo ============================================================
echo.

