@echo off
echo ============================================================
echo STEP 2: GENERATE TRAINING DATA
echo ============================================================
echo.

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Run SETUP.bat first
    pause
    exit /b 1
)

if not exist "data\knowledge_base" (
    echo ERROR: Run 1_GENERATE_KB.bat first
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo Generating training data...
echo - 1500 samples (50 variations per seed)
echo - Split: 1200 train / 150 val / 150 test
echo.

python scripts\generate_training_data_v2.py

if errorlevel 1 (
    echo.
    echo ERROR: Data generation failed
    pause
    exit /b 1
)

echo.
echo ============================================================
echo ✅ STEP 2 COMPLETE!
echo ============================================================
echo.
echo Next: Run 3_BUILD_RAG.bat to build RAG system
echo.
pause
