# CUDA 11.7 Compatibility Fix

## Problem
The AdaManip project was encountering multiple issues:

1. **CUDA compilation error:**
```
RuntimeError: The current installed version of gcc (13.3.0) is greater than the maximum required version by CUDA 11.7 (11.5.0)
```

2. **IsaacGym library error:**
```
ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory
```

These errors occurred because:
- CUDA 11.7 requires GCC version ≤ 11.5.0
- The conda environment had GCC 13.3.0 installed (too new)
- System GCC 8.5.0 was available (compatible)
- IsaacGym couldn't find the Python shared library in the conda environment

## Solution
The fix involves:
1. Setting environment variables to use the system GCC (8.5.0) instead of the conda GCC (13.3.0) for CUDA compilation
2. Adding both `lib` and `lib64` directories to `LD_LIBRARY_PATH` so IsaacGym can find `libpython3.8.so.1.0`

## Usage

### For Fish Shell (Recommended)
```bash
source setup_cuda_env.fish
```

### For Bash Shell
```bash
source setup_cuda_env.sh
```

## What the Scripts Do

1. **Add conda environment to PATH**: Ensures Python, NVCC, and other tools are available
2. **Set compiler variables**: Forces use of system GCC 8.5.0 instead of conda GCC 13.3.0
3. **Set CUDA environment variables**: Configures CUDA_HOME and library paths
4. **Verify setup**: Shows all relevant versions and paths

## Environment Variables Set

- `PATH`: Includes `/ssdstore/azadaia/conda_envs/adamanip/bin`
- `CC`: Set to `/usr/bin/gcc` (system GCC 8.5.0)
- `CXX`: Set to `/usr/bin/g++` (system G++ 8.5.0)
- `CUDA_HOME`: Set to `/ssdstore/azadaia/conda_envs/adamanip`
- `CUDA_ROOT`: Set to `/ssdstore/azadaia/conda_envs/adamanip`
- `LD_LIBRARY_PATH`: Includes both `/conda_envs/adamanip/lib` and `/conda_envs/adamanip/lib64` for Python and CUDA libraries

## Verification

After running the setup script, you should see:
- Python 3.8.20 from the conda environment
- NVCC 11.7.99 from the conda environment
- GCC 8.5.0 from the system (not conda)
- CUDA available in PyTorch

## Tested Packages

The following packages have been successfully compiled and tested:
- ✅ **PointNet++**: `cd Pointnet2_PyTorch/pointnet2_ops_lib/ && pip install -e .`
- ✅ **PyTorch3D**: `cd pytorch3d && pip install -e .`
- ✅ **CUDA availability**: 8 CUDA devices detected

## Troubleshooting

If you encounter issues:

1. **Check GCC version**: `gcc --version` should show 8.5.0, not 13.3.0
2. **Check environment variables**: `echo $CC` should show `/usr/bin/gcc`
3. **Check CUDA**: `nvcc --version` should show CUDA 11.7
4. **Check Python library**: `ldd` on IsaacGym should find `libpython3.8.so.1.0`
5. **Re-run setup**: Source the appropriate script again

### Common Issues and Solutions

**IsaacGym ImportError: libpython3.8.so.1.0**
- **Cause**: `LD_LIBRARY_PATH` doesn't include the conda `lib` directory
- **Solution**: Ensure both `lib` and `lib64` are in `LD_LIBRARY_PATH`
- **Check**: `ls /ssdstore/azadaia/conda_envs/adamanip/lib/libpython3.8*`

**CUDA Compilation Errors**
- **Cause**: Wrong GCC version being used
- **Solution**: Set `CC` and `CXX` to system GCC
- **Check**: `echo $CC` should show `/usr/bin/gcc`

## Compatibility Matrix

| Component | Version | Status |
|-----------|---------|--------|
| CUDA | 11.7.99 | ✅ Compatible |
| System GCC | 8.5.0 | ✅ Compatible |
| Conda GCC | 13.3.0 | ❌ Too new |
| Python | 3.8.20 | ✅ Compatible |
| PyTorch | 1.13.1+cu117 | ✅ Compatible |

## Notes

- Always run the setup script in each new terminal session
- The fix is temporary per session - environment variables reset when terminal closes
- Both bash and fish shell versions are provided
- The system GCC (8.5.0) is older but stable and fully compatible with CUDA 11.7 