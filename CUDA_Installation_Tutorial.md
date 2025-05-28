# Installing Different Versions of NVCC and CUDA Development Tools with Conda - Complete Tutorial

### 1. Check Current CUDA/NVCC Status
```bash
# Check if nvcc is available
which nvcc
nvcc --version

# Check CUDA_HOME environment variable
echo $CUDA_HOME

# Check NVIDIA driver
nvidia-smi
```

### 3. Installation Strategies

```bash
# 1. Create environment
conda create -n dl_env python=3.8
conda activate dl_env

# 2. Install development tools (for compiling extensions)
conda install cudatoolkit-dev=11.7 -c conda-forge

# 3. Install NVCC compiler
conda install cuda-nvcc=11.7 -c nvidia

# 4. Install PyTorch with matching CUDA
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# 5. Now compile extensions (e.g., PointNet++)
cd Pointnet2_PyTorch/pointnet2_ops_lib/
pip install -e .
```

#### Strategy B: All-in-One Approach
```bash
# Install everything at once
conda install cuda-toolkit=11.7 cudatoolkit-dev=11.7 -c nvidia -c conda-forge
```

#### Strategy C: Minimal Installation
```bash
# Install specific CUDA version (full toolkit)
conda install cuda-toolkit=11.7 -c nvidia
```

### 4. Verify Installation
```bash
# Check NVCC version
nvcc --version

# Should output something like:
# Cuda compilation tools, release 11.7, V11.7.64
# Build cuda_11.7.r11.7/compiler.31294372_0

# Check CUDA_HOME (should be set automatically)
echo $CUDA_HOME

# Test CUDA availability in Python
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Available CUDA Versions
Common CUDA versions available through conda:
- **CUDA 10.2**: `cuda-toolkit=10.2`
- **CUDA 11.0**: `cuda-toolkit=11.0`
- **CUDA 11.1**: `cuda-toolkit=11.1`
- **CUDA 11.2**: `cuda-toolkit=11.2`
- **CUDA 11.6**: `cuda-toolkit=11.6`
- **CUDA 11.7**: `cuda-toolkit=11.7`
- **CUDA 11.8**: `cuda-toolkit=11.8`
- **CUDA 12.0**: `cuda-toolkit=12.0`
- **CUDA 12.1**: `cuda-toolkit=12.1`
- **CUDA 12.2**: `cuda-toolkit=12.2`

## Channels to Use
- **nvidia**: Official NVIDIA channel (recommended for toolkit and nvcc)
- **conda-forge**: Community-maintained packages (good for cudatoolkit-dev)
- **defaults**: Default conda channel

## Environment Variables
After installation, these should be automatically set:
```bash
echo $CUDA_HOME          # Should point to conda environment
echo $CUDA_ROOT          # Alternative variable  
echo $LD_LIBRARY_PATH    # Should include CUDA libs
echo $PATH               # Should include CUDA bin directory
```

If not set automatically:
```bash
export CUDA_HOME=$CONDA_PREFIX
export CUDA_ROOT=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib64:$LD_LIBRARY_PATH
export PATH=$CONDA_PREFIX/bin:$PATH
```

## Troubleshooting

### If NVCC is Not Found
```bash
# Add conda environment to PATH
export PATH=$CONDA_PREFIX/bin:$PATH

# Or reinstall with specific channel
conda install cuda-toolkit=11.7 -c nvidia --force-reinstall
```

### Version Conflicts
```bash
# Remove existing CUDA packages
conda remove cuda-toolkit cuda-nvcc cudatoolkit-dev cudatoolkit

# Clean install specific version
conda install cuda-toolkit=11.7 -c nvidia
```

### Development Tools Issues
```bash
# Force reinstall development tools
conda remove cudatoolkit-dev
conda install cudatoolkit-dev=11.7 -c conda-forge --force-reinstall

# Set environment variables manually if needed
export CUDA_HOME=$CONDA_PREFIX
export CUDA_ROOT=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib64:$LD_LIBRARY_PATH
```

### Check Available Versions
```bash
# List available CUDA toolkit versions
conda search cuda-toolkit -c nvidia

# List available NVCC versions
conda search cuda-nvcc -c nvidia

# List available development tools
conda search cudatoolkit-dev -c conda-forge
```

### Compilation Failures
If Python extensions still fail to compile:
```bash
# Ensure all development tools are present
conda install cudatoolkit-dev=11.7 cuda-nvcc=11.7 -c conda-forge -c nvidia

# Check that headers are available
ls $CONDA_PREFIX/include/cuda*

# Check that libraries are available
ls $CONDA_PREFIX/lib/libcuda*
```

## Common Use Cases

### For PointNet++ Installation
```bash
conda create -n pointnet python=3.8
conda activate pointnet
conda install cudatoolkit-dev=11.7 cuda-nvcc=11.7 -c conda-forge -c nvidia
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
cd Pointnet2_PyTorch/pointnet2_ops_lib/
pip install -e .
```

### For PyTorch3D Installation
```bash
conda create -n pytorch3d python=3.8
conda activate pytorch3d
conda install cudatoolkit-dev=11.7 cuda-nvcc=11.7 -c conda-forge -c nvidia
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install pytorch3d
```