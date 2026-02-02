#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate

export TORCH_LIB="$VIRTUAL_ENV/lib/python3.11/site-packages/torch/lib"
export NVIDIA13="$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia/cu13/lib"
export CUBLAS12="$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia/cublas/lib"
export CUDART12="$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia/cuda_runtime/lib"
export LD_LIBRARY_PATH="$TORCH_LIB:$CUBLAS12:$CUDART12:$NVIDIA13:$LD_LIBRARY_PATH"

python wgp.py --listen --share "$@"
