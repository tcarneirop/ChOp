
#!/bin/bash

export CHPL_HOME=~/chapel-1.23.0

export CHPL_HOST_PLATFORM=`$CHPL_HOME/util/chplenv/chpl_platform.py`

export PATH="$PATH":"$CHPL_HOME/bin/$CHPL_HOST_PLATFORM"

export MANPATH="$MANPATH":"$CHPL_HOME"/man



NUM_T_LOCALE=$(cat /proc/cpuinfo | grep processor | wc -l)

export CHPL_RT_NUM_THREADS_PER_LOCALE=$NUM_T_LOCALE
export CHPL_TASKS=qthreads


export fsp=$(pwd)

echo $fsp

echo -e \#\#\# Building runtime 1.23  \#\#\# 

cd $CHPL_HOME
make

cd $fsp





