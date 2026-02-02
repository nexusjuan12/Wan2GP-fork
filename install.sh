#!/bin/bash
# Wan2GP install script for Vast.ai
# Base image: nvidia/cuda:13.1.1-cudnn-devel-ubuntu22.04
# Target: RTX 50XX / Blackwell Pro 6000 (Python 3.11 / PyTorch 2.10 / CUDA 13.0)
#
# Usage:
#   git clone https://github.com/nexusjuan12/Wan2GP-fork.git ~/Wan2GP-fork
#   cd ~/Wan2GP-fork && ./install.sh
set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── System packages ──────────────────────────────────────────────────────────

apt-get update
apt-get install -y --no-install-recommends \
    software-properties-common \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0

# Python 3.11 from deadsnakes PPA
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update
apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv

apt-get clean
rm -rf /var/lib/apt/lists/*

# ── SageAttention (clone next to repo) ───────────────────────────────────────

SAGE_DIR="$REPO_DIR/../SageAttention"
if [ ! -d "$SAGE_DIR" ]; then
    git clone https://github.com/thu-ml/SageAttention.git "$SAGE_DIR"
fi

# ── Python venv ──────────────────────────────────────────────────────────────

if [ ! -d "$REPO_DIR/venv" ]; then
    python3.11 -m venv "$REPO_DIR/venv"
fi
source "$REPO_DIR/venv/bin/activate"

pip install --upgrade pip

# ── PyTorch 2.10 + CUDA 13.0 ────────────────────────────────────────────────

pip install torch==2.10.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu130

# ── Triton ───────────────────────────────────────────────────────────────────

pip install -U triton

# ── SageAttention (build from source) ────────────────────────────────────────

pip install "setuptools<=75.8.2" --force-reinstall
pip install -e "$SAGE_DIR"

# ── ONNX runtime CUDA dependency (cufft for rembg) ──────────────────────────

pip install nvidia-cufft-cu12

# ── Wan2GP requirements ──────────────────────────────────────────────────────

pip install -r "$REPO_DIR/requirements.txt"

# ── Optional kernels (RTX 50XX / Blackwell) ──────────────────────────────────

# lightx2v kernel (Linux, Python 3.11, PyTorch 2.10, CUDA 13)
pip install https://github.com/deepbeepmeep/kernels/releases/download/Light2xv/lightx2v_kernel-0.0.2+torch2.10.0-cp311-abi3-linux_x86_64.whl

echo ""
echo "Install complete. Run with:"
echo "  cd $REPO_DIR && ./run.sh"
