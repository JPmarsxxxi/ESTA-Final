@echo off
REM Secretary Module - Conda Environment Setup (Windows)
REM This creates a conda environment with Python 3.11 for TTS compatibility

echo.
echo ========================================
echo Secretary Module - Conda Setup
echo ========================================
echo.

REM Check if conda is available
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Conda not found! This script requires Anaconda or Miniconda.
    echo Download from: https://www.anaconda.com/download
    pause
    exit /b 1
)

echo [OK] Conda found
conda --version

REM Check if environment exists
conda env list | findstr "esta" >nul 2>&1
if %errorlevel% equ 0 (
    echo.
    echo [WARNING] Conda environment 'esta' already exists.
    set /p recreate="Remove and recreate? (y/n): "
    if /i "%recreate%"=="y" (
        echo Removing existing environment...
        conda env remove -n esta -y
    ) else (
        echo Using existing environment...
        call conda activate esta
        echo [OK] Environment activated
        goto :end
    )
)

echo.
echo [STEP 1/5] Creating conda environment with Python 3.11...
echo This may take a few minutes...
conda create -n esta python=3.11 -y
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create conda environment
    pause
    exit /b 1
)
echo [OK] Conda environment created

echo.
echo [STEP 2/5] Activating environment...
call conda activate esta
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate environment
    pause
    exit /b 1
)
echo [OK] Environment activated

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
if %errorlevel% neq 0 (
    echo [WARNING] TTS installation may have issues. Continuing...
)
echo [OK] All dependencies installed

echo.
echo [STEP 5/5] Checking GPU availability...
python -c "import torch; print('[OK] GPU Available: ' + str(torch.cuda.is_available())); print('    Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')" 2>nul
if %errorlevel% neq 0 (
    echo [INFO] Checking GPU...
    python -c "import torch; print('[OK] GPU:', torch.cuda.is_available())" 2>nul
    if %errorlevel% neq 0 (
        echo [INFO] PyTorch installed, GPU check will be available after restart
    )
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
echo    conda activate esta
echo.
echo 5. Start Jupyter:
echo    jupyter notebook
echo.
echo 6. Open: test_secretary_local.ipynb
echo.
echo NOTE: You're using Python 3.11 in the 'esta' environment
echo       (Python 3.13 is not yet supported by TTS)
echo.
pause
