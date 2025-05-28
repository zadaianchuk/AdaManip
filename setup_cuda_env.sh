#!/bin/bash

# Setup script for CUDA 11.7 compatibility with GCC 8.5.0
# This fixes the "RuntimeError: The current installed version of gcc (13.3.0) is greater than the maximum required version by CUDA 11.7 (11.5.0)"
# Also fixes the "libpython3.8.so.1.0: cannot open shared object file" issue
# Also fixes gymtorch compilation errors

echo "Setting up CUDA 11.7 compatible environment..."

# Add the adamanip conda environment to PATH
export PATH="/ssdstore/azadaia/conda_envs/adamanip/bin:$PATH"

# Set compiler environment variables to use system GCC 8.5.0 (compatible with CUDA 11.7)
# instead of conda GCC 13.3.0 (incompatible)
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

# Set CUDA environment variables
export CUDA_HOME="/ssdstore/azadaia/conda_envs/adamanip"
export CUDA_ROOT="/ssdstore/azadaia/conda_envs/adamanip"
# Include both lib and lib64 directories for Python and CUDA libraries
export LD_LIBRARY_PATH="/ssdstore/azadaia/conda_envs/adamanip/lib:/ssdstore/azadaia/conda_envs/adamanip/lib64:$LD_LIBRARY_PATH"

# Fix gymtorch compilation issues
export MAX_JOBS=1  # Limit parallel compilation to avoid memory issues
export TORCH_CUDA_ARCH_LIST="6.0;6.1+PTX;7.0;7.5+PTX;8.0;8.6+PTX"  # Support all common GPU architectures

# Verify setup
echo "Environment setup complete!"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "NVCC: $(which nvcc)"
echo "NVCC version: $(nvcc --version | grep "release")"
echo "GCC: $(which gcc)"
echo "GCC version: $(gcc --version | head -1)"
echo "CC: $CC"
echo "CXX: $CXX"
echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "MAX_JOBS: $MAX_JOBS"
echo "TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"

# echo ""
# echo "You can now run CUDA compilation commands."
# echo "For example, to compile PointNet++ extensions:"
# echo "cd Pointnet2_PyTorch/pointnet2_ops_lib/ && pip install -e ."
# echo ""
# echo "To clear cached extensions if needed:"
# echo "rm -rf ~/.cache/torch_extensions/py38_cu117/*"