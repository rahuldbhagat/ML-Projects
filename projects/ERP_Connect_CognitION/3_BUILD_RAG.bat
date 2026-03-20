@echo off
echo ============================================================
echo STEP 3: BUILD RAG SYSTEM
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

echo Building RAG system...
echo - Loading documents from knowledge base
echo - Semantic chunking (~287 chunks expected)
echo - Generating embeddings
echo - Building FAISS index
echo.

python scripts\build_rag_v2.py

if errorlevel 1 (
    echo.
    echo ERROR: RAG build failed
    pause
    exit /b 1
)

echo.
echo ============================================================
echo ✅ STEP 3 COMPLETE!
echo ============================================================
echo.
echo Next: Run 4_TRAIN.bat to train the model
echo.
pause
