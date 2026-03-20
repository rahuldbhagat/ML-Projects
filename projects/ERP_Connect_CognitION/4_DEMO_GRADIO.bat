@echo off
echo ============================================================
echo STEP 7: LAUNCH GRADIO DEMO
echo ============================================================
echo.

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Run SETUP.bat first
    pause
    exit /b 1
)

if not exist "models\tinyllama-fine-tuned" (
    echo ERROR: Run 4_TRAIN.bat first
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo.
echo Starting Gradio web interface...
echo.
echo Once started, open your browser to:
echo   http://localhost:7860
echo.
echo Press Ctrl+C to stop the server
echo.

python scripts\inference_gradio.py

pause
