#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Supernan AI — Colab/Kaggle Setup Script
# Run this in the first cell of your notebook:
#   !bash colab/colab_setup.sh
# ═══════════════════════════════════════════════════════════════

set -e

echo "╔═══════════════════════════════════════════════════╗"
echo "║   🚀 Supernan AI — Environment Setup              ║"
echo "╚═══════════════════════════════════════════════════╝"

# ── System Dependencies ─────────────────────────────────────
echo ""
echo "📦 Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq ffmpeg libsndfile1 > /dev/null 2>&1
echo "   ✅ ffmpeg installed"

# ── Python Dependencies ─────────────────────────────────────
echo ""
echo "🐍 Installing Python packages..."
pip install -q --upgrade pip

# Core ML
pip install -q torch torchvision torchaudio
pip install -q numpy tqdm rich click

# Audio
pip install -q pydub librosa soundfile scipy

# Whisper
pip install -q openai-whisper

# Translation
pip install -q transformers sentencepiece protobuf
pip install -q googletrans==4.0.0-rc.1

# Voice Cloning
pip install -q TTS gTTS

# Face Enhancement
pip install -q gfpgan basicsr facexlib realesrgan

# Video/Image
pip install -q opencv-python Pillow imageio imageio-ffmpeg

echo "   ✅ All Python packages installed"

# ── Verify GPU ───────────────────────────────────────────────
echo ""
echo "🖥️  Checking GPU..."
python -c "
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'   ✅ GPU: {gpu} ({mem:.1f} GB)')
else:
    print('   ⚠️  No GPU detected — running on CPU (slower)')
"

# ── Verify Installation ─────────────────────────────────────
echo ""
echo "🔍 Verifying installation..."
python -c "
import whisper
print('   ✅ Whisper')
from transformers import AutoTokenizer
print('   ✅ Transformers')
from TTS.api import TTS
print('   ✅ Coqui TTS')
import cv2
print('   ✅ OpenCV')
print('')
print('╔═══════════════════════════════════════════════════╗')
print('║   ✅ Setup complete! Ready to dub videos.         ║')
print('╚═══════════════════════════════════════════════════╝')
"

echo ""
echo "Usage:"
echo "  python dub_video.py --input video.mp4 --output dubbed.mp4 --start 15 --end 30"
echo ""
