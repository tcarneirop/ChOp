
#!/bin/bash

echo " ### exporting...  ### "

ml cmake/3.23.3_gcc-10.4.0
ml gcc/13.2.0_gcc-10.4.0
#ml hip/5.2.0_gcc-10.4.0

#echo "Setting up LLVM 18 repositories..."
#wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo-g5k apt-key add -
#echo "deb http://apt.llvm.org/bullseye/ llvm-toolchain-bullseye-18 main" | sudo-g5k tee /etc/apt/sources.list.d/llvm18.list

#echo "Installing LLVM 18 and development headers..."
#sudo-g5k apt update
#sudo-g5k apt install -y clang-18 lld-18 libclang-18-dev libomp-18-dev clang-tools-18

sudo-g5k apt-get update
sudo-g5k apt-get install -y hipcc
sudo-g5k apt-get install -y rocprim-dev hipcub-dev
# 3. SET COMPILER VARIABLES
# Switching to the verified LLVM 18 paths
#export LLVM_ROOT=/usr/lib/llvm-18
#export CC=$LLVM_ROOT/bin/clang
#export CXX=$LLVM_ROOT/bin/clang++
export LD_LIBRARY_PATH=$LLVM_ROOT/lib:/opt/rocm-6.3.3/lib:$LD_LIBRARY_PATH


export HIP_PATH=/opt/rocm-6.3.3/
export ROCM_PATH=/opt/rocm-6.3.3/
export DEVICE_LIB_PATH=/opt/rocm-6.3.3/amdgcn/bitcode/
export CHPL_HOME=~/chapel-2.8.0
export CHPL_COMM=none
export CHPL_LLVM=bundled
export CHPL_LOCALE_MODEL=gpu
export CHPL_HOST_PLATFORM=`$CHPL_HOME/util/chplenv/chpl_platform.py`
export CHPL_GPU=amd
export CHOP_HOME=~/ChOp
export CHPL_ROCM_PATH=/opt/rocm-6.3.3/
export CHPL_GPU_ARCH=$(rocm_agent_enumerator | grep -v gfx000 | sort -u | head -1)
export CHPL_GPU_MEM_STRATEGY=array_on_device

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH":"$CHOP_HOME"/libs

CHPL_BIN_SUBDIR=`"$CHPL_HOME"/util/chplenv/chpl_bin_subdir.py`
export PATH="$PATH":"$CHPL_HOME/bin/$CHPL_BIN_SUBDIR"

echo $LD_LIBRARY_PATH



export MANPATH="$MANPATH":"$CHPL_HOME"/man

NUM_T_LOCALE=$(cat /proc/cpuinfo | grep processor | wc -l)

export CHPL_TARGET_ARCH=native
NUM_T_LOCALE=$(cat /proc/cpuinfo | grep processor | wc -l)
export CHPL_RT_NUM_THREADS_PER_LOCALE=$NUM_T_LOCALE
#export CHPL_RT_NUM_THREADS_PER_LOCALE=1
export CHPL_TASKS=qthreads

echo -e \#\#\#QThreads set for $NUM_T_LOCALE threads\#\#\#.

export here=$(pwd)

echo $here


cd $CHPL_HOME
make -j $NUM_T_LOCALE

echo -e \#\#\# Building runtime 2.80 AMD GPU - Single Locale  \#\#\#

cd $here

