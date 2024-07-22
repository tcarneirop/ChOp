
#!/bin/bash

echo " ### exporting...  ### "

export CHPL_HOME=/scratch/carnei26/chapel-2.1.0

export CHPL_LLVM=system
#export CHPL_LLVM=none
export CHOP_HOME=/scratch/carnei26/ChOp
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH":"$CHOP_HOME"/libs

CHPL_BIN_SUBDIR=`"$CHPL_HOME"/util/chplenv/chpl_bin_subdir.py`
export PATH="$PATH":"$CHPL_HOME/bin/$CHPL_BIN_SUBDIR"

echo $LD_LIBRARY_PATH

export CHPL_HOST_PLATFORM=`$CHPL_HOME/util/chplenv/chpl_platform.py`

export MANPATH="$MANPATH":"$CHPL_HOME"/man

NUM_T_LOCALE=$(cat /proc/cpuinfo | grep processor | wc -l)

export CHPL_TARGET_ARCH=native
export CHPL_RT_NUM_THREADS_PER_LOCALE=$NUM_T_LOCALE
export CHPL_TASKS=qthreads

echo -e \#\#\#QThreads set for $NUM_T_LOCALE threads\#\#\#.

export here=$(pwd)

echo $here


cd $CHPL_HOME
make -j ${NUM_T_LOCALE}

echo -e \#\#\# Building runtime v2.1  QTHREADS.  \#\#\#

cd $here
