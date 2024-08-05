#!/bin/bash -l
module unload CUDA

module load CMake/3.16.2
module load AOCC/3.1.0-GCCcore-10.3.0
module load CUDA/11.3.1

export  GASNET_PHYSMEM_MAX='20 GB'

# Setup Chapel Home
export CHPL_VERSION="1.30.0"
export CHPL_TARGET_CPU=native
# Put and compile chapel next to the launcher
export CHPL_HOME="~/chapel-${CHPL_VERSION}"
export CHPL_LAUNCHER="gasnetrun_ibv"
# Classical linux64 platform
export CHPL_HOST_PLATFORM="linux64"
# Use third-party LLVM
export CHPL_LLVM='none'
# Number of threads on each local

NUM_T_LOCALE=$(cat /proc/cpuinfo | grep processor | wc -l)

export CHPL_RT_NUM_THREADS_PER_LOCALE=$NUM_T_LOCALE

##I need to think how to use infiniband...
export CHPL_COMM='gasnet'
export CHPL_COMM_SUBSTRATE='ibv'
export CHPL_TARGET_CPU='native'
export GASNET_QUIET=1
export HFI_NO_CPUAFFINITY=1

# Specify ssh spawner
export GASNET_IBV_SPAWNER='ssh'

#if [ ! -d "$CHPL_HOME" ]; then
    #wget -c https://github.com/chapel-lang/chapel/releases/download/1.25.0/chapel-${CHPL_VERSION}.tar.gz -O - | tar xz
cd $CHPL_HOME
make -j ${CHPL_RT_NUM_THREADS_PER_LOCALE}
cd ..
#fi

CHPL_BIN_SUBDIR=`"$CHPL_HOME"/util/chplenv/chpl_bin_subdir.py`
export PATH="$PATH":"$CHPL_HOME/bin/$CHPL_BIN_SUBDIR:$CHPL_HOME/util"

export CHOP_HOME="/home/it4i-tcarn/ChOp"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"$CHOP_HOME/libs"

echo $LD_LIBRARY_PATH


# Start computation
export GASNET_SSH_SERVERS=$(cat $PBS_NODEFILE | xargs echo)
echo $GASNET_SSH_SERVERS
#chpl -o hello $CHPL_HOME/examples/hello6-taskpar-dist.chpl

#./hello -nl 2 > hello2

#cd $CHOP_HOME

#make
#./bin/chop.out --size=20 --lower_bound="queens" --scheduler="dynamic" --mlsearch="mlgpu" --coordinated=false --pgas=false --initial_depth=2 --second_depth=4 --mlchunk=1 --lchunk=1 -nl 2 > mlmgpu2024
#./bin/chop.out --size=20 --lower_bound="queens" --scheduler="dynamic" --mlsearch="mlgpu" --coordinated=false --pgas=false --initial_depth=2 --second_depth=5 --mlchunk=1 --lchunk=1 -nl 2 > mlmgpu2025
#./bin/chop.out --size=20 --lower_bound="queens" --scheduler="dynamic" --mlsearch="mlgpu" --coordinated=false --pgas=false --initial_depth=2 --second_depth=6 --mlchunk=1 --lchunk=1 -nl 2 > mlmgpu2026
#./bin/chop.out --size=21 --lower_bound="queens" --scheduler="dynamic" --mlsearch="mlgpu" --coordinated=false --pgas=false --initial_depth=2 --second_depth=4 --mlchunk=1 --lchunk=1 -nl 2 > mlmgpu2124
#./bin/chop.out --size=21 --lower_bound="queens" --scheduler="dynamic" --mlsearch="mlgpu" --coordinated=false --pgas=false --initial_depth=2 --second_depth=5 --mlchunk=1 --lchunk=1 -nl 2 > mlmgpu2125
#./bin/chop.out --size=21 --lower_bound="queens" --scheduler="dynamic" --mlsearch="mlgpu" --coordinated=false --pgas=false --initial_depth=2 --second_depth=6 --mlchunk=1 --lchunk=1 -nl 2 > mlmgpu2126


