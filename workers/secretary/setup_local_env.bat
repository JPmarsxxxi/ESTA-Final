@echo off
REM Secretary Module - Windows Local Environment Setup Script
REM This creates an isolated Python environment for the Secretary module

echo.
echo ========================================
echo Secretary Module - Environment Setup
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! Please install Python 3.8+ first.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [OK] Python found
python --version

REM Check if venv exists
if exist "venv\" (
    echo.
    echo [WARNING] Virtual environment already exists.
    set /p recreate="Remove and recreate? (y/n): "
    if /i "%recreate%"=="y" (
        echo Removing existing venv...
        rmdir /s /q venv
    ) else (
        echo Using existing venv...
        call venv\Scripts\activate.bat
        echo [OK] Virtual environment activated
        goto :end
    )
)

echo.
echo [STEP 1/5] Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment created

echo.
echo [STEP 2/5] Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment activated

echo.
echo [STEP 3/5] Upgrading pip...
python -m pip install --upgrade pip setuptools wheel --quiet
echo [OK] Pip upgraded

echo.
echo [STEP 4/5] Installing dependencies...
echo    - Jupyter and widgets...
pip install jupyter notebook ipywidgets --quiet
echo    - Secretary requirements...
pip install -r requirements.txt --quiet
echo    - TTS and Whisper (this may take 5-10 minutes)...
pip install TTS openai-whisper --quiet
echo [OK] All dependencies installed

echo.
echo [STEP 5/5] Checking GPU availability...
python -c "import torch; print('[OK] GPU Available: ' + str(torch.cuda.is_available())); print('    Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')" 2>nul
if %errorlevel% neq 0 (
    echo [INFO] PyTorch installed, checking GPU...
    python -c "import torch; print('[OK] GPU:', torch.cuda.is_available())"
)

:end
echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo NEXT STEPS:
echo.
echo 1. Install Ollama for Windows:
echo    Download from: https://ollama.com/download/windows
echo    Or run: winget install Ollama.Ollama
echo.
echo 2. Start Ollama (it usually starts automatically)
echo    Check system tray for Ollama icon
echo.
echo 3. Pull Mistral model (run in new terminal):
echo    ollama pull mistral
echo.
echo 4. Activate environment (whenever you start new terminal):
echo    venv\Scripts\activate.bat
echo.
echo 5. Start Jupyter:
echo    jupyter notebook
echo.
echo 6. Open: test_secretary_local.ipynb
echo.
pause
