#!/bin/bash
set -e

# Variables
ENV_NAME="wan2gp"
PYTHON_VERSION="3.10.9"
WAN_REPO="https://github.com/nexusjuan12/Wan2GP-fork.git"
SAGE_REPO="https://github.com/thu-ml/SageAttention.git"

echo "[1] Installing system dependencies..."
sudo apt update
sudo apt install -y git wget build-essential ninja-build libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg

echo "[2] Installing Miniconda..."
cd ~
wget -O Miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py310_24.1.2-0-Linux-x86_64.sh
bash Miniconda.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

echo "[3] Creating Conda environment: $ENV_NAME"
conda create -y -n $ENV_NAME python=$PYTHON_VERSION
conda activate $ENV_NAME

echo "[4] Installing PyTorch 2.7 (nightly) with CUDA 12.8"
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

echo "[5] Cloning Wan2GP..."
git clone "$WAN_REPO"
cd Wan2GP-fork

echo "[6] Installing WAN2GP dependencies..."
pip install -r blackwell-requirements.txt

echo "[7] Installing Triton and compiling SageAttention..."
cd ~
pip install triton==2.1.0
git clone "$SAGE_REPO"
cd SageAttention
pip install -e .
cd ~

echo "[8] DONE — To launch WAN2GP:"
echo "  conda activate $ENV_NAME"
echo "  cd Wan2GP"
echo "  python wgp.py --t2v-14B --attention sage --share"
