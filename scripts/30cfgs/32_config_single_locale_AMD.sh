
#!/bin/bash

echo " ### exporting...  ### "

export CHPL_HOME=~/chapel-1.32.0
export CHPL_COMM=none
#export CHPL_LLVM=system
export CHPL_LLVM=bundled
export CHPL_LOCALE_MODEL=gpu
export CHPL_HOST_PLATFORM=`$CHPL_HOME/util/chplenv/chpl_platform.py`
export CHPL_GPU=amd
export CHOP_HOME=~/ChOp
export CHPL_ROCM_PATH=/opt/rocm-4.5.0
export CHPL_GPU_ARCH=gfx906

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
make -j 32

echo -e \#\#\# Building runtime 1.31  GPU.  \#\#\#

cd $here
