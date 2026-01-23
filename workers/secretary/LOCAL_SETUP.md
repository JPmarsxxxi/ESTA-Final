# Secretary Module - Local Setup Guide

Run the Secretary module locally on your machine with Jupyter Notebook.

## Important: Python Version Compatibility

**TTS (Coqui TTS) requires Python 3.11 or 3.12.** It does NOT support Python 3.13+.

### If you have Anaconda/Miniconda (RECOMMENDED):
Use `setup_conda_env.bat` (Windows) or `setup_conda_env.sh` (Linux/Mac) to create an environment with Python 3.11.

### If you have standalone Python:
- **Python 3.11 or 3.12**: Use `setup_local_env.bat` / `setup_local_env.sh`
- **Python 3.13+**: You must install Python 3.11 or 3.12 first, or use Conda

## Quick Start

### Option A: Conda Setup (Recommended for Anaconda users)

**Windows:**
```cmd
cd workers\secretary
setup_conda_env.bat
conda activate esta
ollama pull mistral
jupyter notebook
```

**Linux/Mac:**
```bash
cd workers/secretary
./setup_conda_env.sh
conda activate esta
ollama pull mistral
jupyter notebook
```

### Option B: Virtual Environment (Python 3.11-3.12 only)

### Windows

```cmd
cd workers\secretary

REM 1. Setup environment (creates esta, installs dependencies)
setup_local_env.bat

REM 2. Install Ollama for Windows
REM Download from: https://ollama.com/download/windows
REM Or: winget install Ollama.Ollama

REM 3. Pull Mistral model (Ollama starts automatically on Windows)
ollama pull mistral

REM 4. Activate environment (when starting new terminal)
esta\Scripts\activate.bat

REM 5. Start Jupyter
jupyter notebook

REM 6. Open test_secretary_local.ipynb
```

### Linux / Mac

```bash
cd workers/secretary

# 1. Setup environment (creates esta, installs dependencies)
./setup_local_env.sh

# 2. Activate environment
source esta/bin/activate

# 3. Install Ollama (if not installed)
curl -fsSL https://ollama.com/install.sh | sh

# 4. Start Ollama (in separate terminal)
ollama serve

# 5. Pull Mistral model (first time only, ~4GB download)
ollama pull mistral

# 6. Start Jupyter
jupyter notebook

# 7. Open test_secretary_local.ipynb
```

## System Requirements

**Required:**
- Python 3.11 or 3.12 (TTS does NOT support 3.13+)
- 8GB+ RAM
- 10GB free disk space

**Recommended:**
- NVIDIA GPU with CUDA support (for faster TTS/Whisper)
- 16GB+ RAM
- SSD storage

## Environment Structure

```
workers/secretary/
├── esta/                          # Environment folder (venv or conda)
├── setup_conda_env.bat            # Conda setup with Python 3.11 (Windows) - RECOMMENDED
├── setup_conda_env.sh             # Conda setup with Python 3.11 (Linux/Mac) - RECOMMENDED
├── setup_local_env.bat            # venv setup (Windows, requires Python 3.11-3.12)
├── setup_local_env.sh             # venv setup (Linux/Mac, requires Python 3.11-3.12)
├── test_secretary_local.ipynb     # Interactive testing notebook
├── secretary.py                   # Main orchestrator
├── tool_registry.py               # Worker registry
├── scriptwriter.py                # Real ScriptWriter (Ollama + LangSearch)
├── audio_agent.py                 # Real AudioAgent (XTTS + Whisper)
├── requirements.txt               # Python dependencies
└── audio_outputs/                 # Generated audio files (created on first run)
```

## What Gets Installed

**Python Packages:**
- `jupyter` - Notebook interface
- `requests` - HTTP client
- `TTS` - Coqui TTS for voice generation (~1.5GB)
- `openai-whisper` - Speech transcription (~1GB)
- `torch` - Deep learning framework (GPU or CPU)

**External Services:**
- `ollama` - Local LLM server (separate install)
- `mistral` model - 7B parameter LLM (~4GB)

## GPU vs CPU

**GPU (NVIDIA CUDA):**
- Script generation: 30-60s
- Audio generation: 1-3 minutes
- Total: ~2-4 minutes per video

**CPU Only:**
- Script generation: 30-60s (same, Ollama is efficient)
- Audio generation: 5-10 minutes (TTS is slower)
- Total: ~6-11 minutes per video

The notebook will auto-detect your GPU and use it if available.

## Troubleshooting

### "Ollama: Not running"

**Windows:** Check system tray for Ollama icon. If not running, start from Start menu.

**Linux/Mac:** Start in a separate terminal:
```bash
ollama serve
```

### "Mistral: False"
Pull the model:
```bash
ollama pull mistral
```

### "Not in esta"

**Conda (Recommended):**
```bash
conda activate esta
```

**venv - Windows:**
```cmd
esta\Scripts\activate.bat
```

**venv - Linux/Mac:**
```bash
source esta/bin/activate
```

### "ScriptWriter: MOCK"
Check Ollama is running and Mistral is pulled.

### "AudioAgent: MOCK"
Install TTS and Whisper:
```bash
pip install TTS openai-whisper
```

### Python 3.13+ Installation Error
If you see `ERROR: No matching distribution found for TTS`:
- TTS requires Python 3.11 or 3.12
- **Solution:** Use `setup_conda_env.bat` or `setup_conda_env.sh` to create an environment with Python 3.11
- Or manually install Python 3.11/3.12 and use that instead

### GPU not detected but you have one

**Windows:** Install CUDA toolkit from NVIDIA, then:
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Linux:** Install CUDA drivers and PyTorch with CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Output Files

**Audio files:** `./audio_outputs/audio_<timestamp>.wav`
**Temp files:** `./audio_outputs/temp_<timestamp>.wav`

Files are saved locally and can be played with any audio player.

## Next Steps

After verifying ScriptWriter and AudioAgent work:
1. Build LangSearch worker (web research)
2. Build BrainBox worker (video planning)
3. Build AssetCollector worker (asset gathering)
4. Build Executor worker (final assembly)

Each worker will be built and tested incrementally in the notebook.

## Clean Restart

To completely reset the environment:

**Windows:**
```cmd
rmdir /s /q esta audio_outputs
setup_local_env.bat
```

**Linux/Mac:**
```bash
rm -rf esta audio_outputs
./setup_local_env.sh
```

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify system requirements
3. Ensure all setup steps completed successfully
4. Check notebook cell 1 (Environment Check) for specific errors
