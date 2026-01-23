# Secretary Module - Local Setup Guide

Run the Secretary module locally on your machine with Jupyter Notebook.

## Quick Start

```bash
cd workers/secretary

# 1. Setup environment (creates venv, installs dependencies)
./setup_local_env.sh

# 2. Activate environment
source venv/bin/activate

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
- Python 3.8+
- 8GB+ RAM
- 10GB free disk space

**Recommended:**
- NVIDIA GPU with CUDA support (for faster TTS/Whisper)
- 16GB+ RAM
- SSD storage

## Environment Structure

```
workers/secretary/
├── venv/                          # Virtual environment (created by setup script)
├── setup_local_env.sh             # One-command setup
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
Start Ollama in a separate terminal:
```bash
ollama serve
```

### "Mistral: False"
Pull the model:
```bash
ollama pull mistral
```

### "Not in venv"
Activate the environment:
```bash
source venv/bin/activate
```

### "ScriptWriter: MOCK"
Check Ollama is running and Mistral is pulled.

### "AudioAgent: MOCK"
Install TTS and Whisper:
```bash
pip install TTS openai-whisper
```

### GPU not detected but you have one
Install CUDA drivers and PyTorch with CUDA:
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
```bash
rm -rf venv audio_outputs
./setup_local_env.sh
```

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify system requirements
3. Ensure all setup steps completed successfully
4. Check notebook cell 1 (Environment Check) for specific errors
