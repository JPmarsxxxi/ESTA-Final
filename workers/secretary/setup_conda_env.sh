#!/bin/bash

# Secretary Module - Conda Environment Setup (Linux/Mac)
# This creates a conda environment with Python 3.11 for TTS compatibility

set -e

echo "========================================="
echo "Secretary Module - Conda Setup"
echo "========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found! This script requires Anaconda or Miniconda."
    echo "Download from: https://www.anaconda.com/download"
    exit 1
fi

echo "✓ Conda found"
conda --version

# Check if environment exists
if conda env list | grep -q "^esta "; then
    echo ""
    echo "⚠️  Conda environment 'esta' already exists."
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n esta -y
    else
        echo "Using existing environment..."
        conda activate esta
        echo "✓ Environment activated"
        exit 0
    fi
fi

echo ""
echo "📦 [STEP 1/5] Creating conda environment with Python 3.11..."
echo "This may take a few minutes..."
conda create -n esta python=3.11 -y

echo ""
echo "✓ [STEP 2/5] Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate esta

echo ""
echo "📥 [STEP 3/5] Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

echo ""
echo "📥 [STEP 4/5] Installing dependencies..."
echo "   - Jupyter and widgets..."
pip install jupyter notebook ipywidgets
echo "   - Secretary requirements..."
pip install -r requirements.txt
echo "   - TTS and Whisper (this may take 5-10 minutes)..."
pip install TTS openai-whisper

echo ""
echo "🔍 [STEP 5/5] Checking GPU availability..."
python -c "import torch; print('✓ GPU Available:', torch.cuda.is_available()); print('   Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')" 2>/dev/null || echo "ℹ️  GPU check will be available after restart"

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "📝 Next steps:"
echo ""
echo "1. Install Ollama (if not already installed):"
echo "   curl -fsSL https://ollama.com/install.sh | sh"
echo ""
echo "2. Start Ollama service:"
echo "   ollama serve"
echo ""
echo "3. Pull Mistral model (in another terminal):"
echo "   ollama pull mistral"
echo ""
echo "4. Activate the environment:"
echo "   conda activate esta"
echo ""
echo "5. Start Jupyter:"
echo "   jupyter notebook"
echo ""
echo "6. Open: test_secretary_local.ipynb"
echo ""
echo "NOTE: You're using Python 3.11 in the 'esta' environment"
echo "      (Python 3.13 is not yet supported by TTS)"
echo ""
