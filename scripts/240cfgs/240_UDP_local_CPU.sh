#!/bin/bash



echo " ### exporting...  ### "

export CHPL_HOME=~/chapel-2.5.0
export CHOP_HOME=~/ChOp
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH":"$CHOP_HOME"/libs

echo $LD_LIBRARY_PATH


CHPL_BIN_SUBDIR=`"$CHPL_HOME"/util/chplenv/chpl_bin_subdir.py`

export PATH="$PATH":"$CHPL_HOME/bin/$CHPL_BIN_SUBDIR"

export CHPL_HOST_PLATFORM=`$CHPL_HOME/util/chplenv/chpl_platform.py`

export MANPATH="$MANPATH":"$CHPL_HOME"/man

source $CHPL_HOME/util/setchplenv.bash


NUM_T_LOCALE=$(cat /proc/cpuinfo | grep processor | wc -l)

export CHPL_COMM=gasnet
export CHPL_COMM_SUBSTRATE=udp
export CHPL_RT_NUM_THREADS_PER_LOCALE=$NUM_T_LOCALE
export CHPL_TASKS=qthreads
export CHPL_LLVM=none
export GASNET_SPAWNFN=L
export CHPL_TARGET_CPU=native

echo -e \#\#\#QThreads set for $NUM_T_LOCALE threads\#\#\#.

export here=$(pwd)

echo $here


pushd $CHPL_HOME
nice make -j `nproc`
popd

echo -e \#\#\# Building runtime 2.5.0  QTHREADS, UDP, and Local spawn.  \#\#\#

cd $here





