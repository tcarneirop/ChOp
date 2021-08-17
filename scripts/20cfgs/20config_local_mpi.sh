#!/bin/bash

export CHPL_HOME=~/chapel/chapel-1.20.0

export CHPL_HOST_PLATFORM=`$CHPL_HOME/util/chplenv/chpl_platform.py`

export PATH="$PATH":"$CHPL_HOME/bin/$CHPL_HOST_PLATFORM"

export MANPATH="$MANPATH":"$CHPL_HOME"/man



NUM_T_LOCALE=$(cat /proc/cpuinfo | grep processor | wc -l)

export CHPL_TARGET_ARCH=native
export CHPL_COMM=gasnet
export CHPL_COMM_SUBSTRATE=mpi
export GASNET_SPAWNFN=L
export CHPL_RT_NUM_THREADS_PER_LOCALE=$NUM_T_LOCALE
export CHPL_TASKS=qthreads

echo -e \#\#\#QThreads set for $NUM_T_LOCALE threads\#\#\#.

export here=$(pwd)

echo $here



cd $CHPL_HOME
make

echo -e \#\#\# Building runtime 1.19 for QTHREADS, MPI, and Local spawn.  \#\#\#

cd $here

rm ../fsp.out

#./compile.sh



