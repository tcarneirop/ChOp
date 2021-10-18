#!/bin/bash


echo " ### exporting...  ### "


export CHPL_HOME=~/chapel-1.25.0

#export CHPL_LLVM=system
export CHPL_LLVM=bundled
export CHOP_HOME=~/ChOp/ChOp
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH":"$CHOP_HOME"/libs

CHPL_BIN_SUBDIR=`"$CHPL_HOME"/util/chplenv/chpl_bin_subdir.py`
export PATH="$PATH":"$CHPL_HOME/bin/$CHPL_BIN_SUBDIR"


echo $LD_LIBRARY_PATH

export CHPL_HOST_PLATFORM=`$CHPL_HOME/util/chplenv/chpl_platform.py`

export MANPATH="$MANPATH":"$CHPL_HOME"/man


NUM_T_LOCALE=$(cat /proc/cpuinfo | grep processor | wc -l)

export CHPL_TARGET_ARCH=native
export CHPL_COMM=gasnet
export CHPL_COMM_SUBSTRATE=udp
export GASNET_SPAWNFN=L
export CHPL_RT_NUM_THREADS_PER_LOCALE=$NUM_T_LOCALE
export CHPL_TASKS=qthreads

echo -e \#\#\#QThreads set for $NUM_T_LOCALE threads\#\#\#.

export here=$(pwd)

echo $here


cd $CHPL_HOME
make

echo -e \#\#\# Building runtime 1.25  QTHREADS, UDP, and Local spawn.  \#\#\#

cd $here





