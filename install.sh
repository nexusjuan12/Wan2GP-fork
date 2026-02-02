#!/bin/bash
# Wan2GP install script for Vast.ai
# Base image: nvidia/cuda:13.1.1-cudnn-devel-ubuntu22.04
# Target: RTX 50XX with FP4 (Python 3.11 / PyTorch 2.10 / CUDA 13.0)
set -e

WORKSPACE="/root/workspace"

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

# ── Clone repos ──────────────────────────────────────────────────────────────

cd "$WORKSPACE"

if [ ! -d "$WORKSPACE/Wan2GP" ]; then
    git clone https://github.com/deepbeepmeep/Wan2GP.git
fi

if [ ! -d "$WORKSPACE/SageAttention" ]; then
    git clone https://github.com/thu-ml/SageAttention.git
fi

# ── Python venv ──────────────────────────────────────────────────────────────

cd "$WORKSPACE/Wan2GP"

if [ ! -d venv ]; then
    python3.11 -m venv venv
fi
source venv/bin/activate

pip install --upgrade pip

# ── PyTorch 2.10 + CUDA 13.0 ────────────────────────────────────────────────

pip install torch==2.10.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu130

# ── Triton ───────────────────────────────────────────────────────────────────

pip install -U triton

# ── SageAttention (build from source) ────────────────────────────────────────

pip install "setuptools<=75.8.2" --force-reinstall
pip install -e "$WORKSPACE/SageAttention"

# ── Wan2GP requirements ──────────────────────────────────────────────────────

pip install -r "$WORKSPACE/Wan2GP/requirements.txt"

# ── Optional kernels (RTX 50XX) ──────────────────────────────────────────────

# lightx2v kernel (Linux, Python 3.11, PyTorch 2.10, CUDA 13)
pip install https://github.com/deepbeepmeep/kernels/releases/download/Light2xv/lightx2v_kernel-0.0.2+torch2.10.0-cp311-abi3-linux_x86_64.whl

echo ""
echo "Install complete. Run with:"
echo "  cd $WORKSPACE/Wan2GP && ./run.sh"
