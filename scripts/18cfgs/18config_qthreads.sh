#!/bin/bash

export CHPL_HOME=~/chapel/chapel-1.18.0

export CHPL_HOST_PLATFORM=`$CHPL_HOME/util/chplenv/chpl_platform.py`

export PATH="$PATH":"$CHPL_HOME/bin/$CHPL_HOST_PLATFORM"

export MANPATH="$MANPATH":"$CHPL_HOME"/man





NUM_T_LOCALE=$(cat /proc/cpuinfo | grep processor | wc -l)

export CHPL_RT_NUM_THREADS_PER_LOCALE=$NUM_T_LOCALE
export CHPL_TASKS=qthreads

echo -e \#\#\#Massivethreads set for $NUM_T_LOCALE threads\#\#\#.

export fsp=$(pwd)

echo $fsp

echo -e \#\#\# Building runtime for Massivethreads. \#\#\# 

cd $CHPL_HOME
make

echo -e \#\#\# Compilling FSP - Massivethreads. \#\#\# 

cd $fsp

rm ../fsp.out

#./compile.sh

#echo -e \#\#\# Running N-Queens - Massivethreads. \#\#\# 

#./queens.out --num_threads=8 --mode="mcore" --size=10 --initial_depth=5 --scheduler="dynamic" --chunk=16
