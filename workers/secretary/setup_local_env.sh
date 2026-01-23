#!/bin/bash

# Secretary Module - Local Environment Setup Script
# This creates an isolated Python environment for the Secretary module

set -e

echo "🚀 Setting up Secretary Module local environment..."

# Check Python version
python_version=$(python3 --version | cut -d' ' -f2)
echo "✓ Python version: $python_version"

# Check if Python 3.13+ (TTS requires 3.11 or 3.12)
python_minor=$(echo $python_version | cut -d'.' -f2)
if [ "$python_minor" -ge 13 ]; then
    echo ""
    echo "⚠️  WARNING: Python 3.13+ detected!"
    echo "TTS library requires Python 3.11 or 3.12."
    echo ""
    echo "RECOMMENDED: Use setup_conda_env.sh instead"
    echo "This will create an environment with Python 3.11."
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Setup cancelled."
        exit 0
    fi
fi

# Create virtual environment
if [ -d "esta" ]; then
    echo "⚠️  Virtual environment already exists. Remove 'esta' folder to recreate."
    read -p "Do you want to remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf esta
    else
        echo "Using existing esta..."
        source esta/bin/activate
        echo "✓ Activated existing virtual environment"
        exit 0
    fi
fi

echo "📦 Creating virtual environment..."
python3 -m venv esta

echo "✓ Activating virtual environment..."
source esta/bin/activate

echo "📥 Installing base dependencies..."
pip install --upgrade pip setuptools wheel

echo "📥 Installing Jupyter and widgets..."
pip install jupyter notebook ipywidgets

echo "📥 Installing Secretary module dependencies..."
pip install -r requirements.txt

echo "📥 Installing TTS and Whisper (this may take a few minutes)..."
pip install TTS openai-whisper

echo ""
echo "✅ Virtual environment setup complete!"
echo ""
echo "📝 Next steps:"
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
echo "   source esta/bin/activate"
echo ""
echo "5. Start Jupyter:"
echo "   jupyter notebook"
echo ""
echo "6. Open: test_secretary_local.ipynb"
echo ""
