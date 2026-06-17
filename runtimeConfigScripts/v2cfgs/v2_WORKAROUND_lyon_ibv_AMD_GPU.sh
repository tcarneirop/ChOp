#!/bin/bash
# ==============================================================================
# v3_GUIX_PURE_TOOLCHAIN_AMD_GPU.sh
# Using full Guix 14.3.0 toolchain to match modern ROCm 7.1.1
# ==============================================================================

echo "### Purging environment variables to avoid host pollution... ###"
unset LD_LIBRARY_PATH LIBRARY_PATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH CPATH
unset CHPL_MAKE_HOST_CFLAGS CHPL_MAKE_HOST_CXXFLAGS CHPL_MAKE_HOST_LDFLAGS CHPL_TARGET_LDFLAGS
unset LDFLAGS CC CXX

unset MODULEPATH
module use /grid5000/guix-modules/x86_64/latest

echo "### Loading PURE Guix Stack (GCC 14 + ROCm 7)... ###"
ml python/3.12.12
ml cmake/4.1.3
ml gcc-toolchain/14.3.0  # Brings in modern GCC + matching modern GLIBC runtime

ml rocm-comgr/7.1.1
ml rocminfo/7.1.1
ml rocm-smi-lib-bin/7.1.1
ml rocm-cmake/7.1.1
ml rocm-hip-runtime/7.1.1
ml rocm-smi-lib/7.1.1
ml rocm-opencl-runtime/7.1.1
ml rocm-toolchain/7.1.1
ml rocm-hipcc/7.1.1

# ==============================================================================
# ROCM PATH MAPPING (Guix Store Layout)
# ==============================================================================
export ROCM_PATH=$(find /gnu/store/ -maxdepth 1 -type d -name "*-rocm-hipcc-7.1.1" | head -n 1)
export HIP_PATH=$ROCM_PATH
export CHPL_ROCM_PATH=$ROCM_PATH

export ROCM_RUNTIME_PATH=$(find /gnu/store/ -maxdepth 1 -type d -name "*-rocm-hip-runtime-7.1.1" | head -n 1)
export DEVICE_LIB_PATH=$(find /gnu/store/ -maxdepth 1 -type d -name "*-rocm-device-libs-7.1.1" | head -n 1)/amdgcn/bitcode
export HIP_DEVICE_LIB_PATH=$DEVICE_LIB_PATH

# ==============================================================================
# CORE CHAPEL ENVIRONMENT SETUP
# ==============================================================================
export CHPL_LLVM=bundled
export CHPL_HOME="$HOME/chapel-2.9.0"
export CHPL_COMM=gasnet
export CHPL_COMM_SUBSTRATE=ibv
export CHPL_TARGET_ARCH=native
export CHPL_GPU=amd
export CHPL_GPU_ARCH=gfx908
export CHPL_CUDA_CFLAGS="-DIMPROVED"

# ==============================================================================
# PRISTINE CLEAN & BUILD
# ==============================================================================
echo "### Wiping previous build state completely... ###"
cd "$CHPL_HOME"
make clean

echo "### Starting bundled compilation sequence... ###"
make -j$(nproc)