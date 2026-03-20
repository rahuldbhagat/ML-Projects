@echo off
echo ============================================================
echo STEP 1: GENERATE KNOWLEDGE BASE
echo ============================================================
echo.

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found
    echo Please run SETUP.bat first
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo Generating synthetic knowledge base...
echo - 25 documents
echo - Architecture, Integration, SAP, HTTP Errors, Performance
echo - ~23,000 words total
echo.

python scripts\generate_synthetic_kb.py

if errorlevel 1 (
    echo.
    echo ERROR: Knowledge base generation failed
    pause
    exit /b 1
)

echo.
echo ============================================================
echo ✅ STEP 1 COMPLETE!
echo ============================================================
echo.
echo Knowledge base created in: data/knowledge_base/
echo.
echo Next: Run 2_GENERATE_DATA.bat to create training samples
echo.
pause
