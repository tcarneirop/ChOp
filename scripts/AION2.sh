#!/bin/bash -l 
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --exclusive


# Helpers
# https://chapel-lang.org/docs/usingchapel/launcher.html
# https://chapel-lang.org/docs/platforms/infiniband.html#using-mpi-for-job-launch
# http://www.hpc-carpentry.org/hpc-chapel/21-locales/index.html
# https://chapel-lang.org/docs/platforms/infiniband.html

# Load the foss toolchain to get access to gcc,mpi,etc...
module load toolchain/foss/2020b 


# Setup Chapel Home
export CHPL_VERSION="1.25.0"
# Put and compile chapel next to the launcher
export CHPL_HOME="${PWD}/chapel-${CHPL_VERSION}"
export CHPL_LAUNCHER="none"
# Classical linux64 platform
export CHPL_HOST_PLATFORM="linux64"
# Use third-party LLVM
export CHPL_LLVM=none
# Number of threads on each local
export CHPL_RT_NUM_THREADS_PER_LOCALE=${SLURM_CPUS_PER_TASK}

##I need to think how to use infiniband...
export CHPL_COMM='gasnet'
export CHPL_COMM_SUBSTRATE='ibv'
export CHPL_TARGET_CPU='native'
export GASNET_QUIET=1
export HFI_NO_CPUAFFINITY=1

# Specify ssh spawner
export GASNET_IBV_SPAWNER=ssh

if [ ! -d "$CHPL_HOME" ]; then
    module load devel/CMake/3.18.4-GCCcore-10.2.0
    wget -c https://github.com/chapel-lang/chapel/releases/download/1.25.0/chapel-${CHPL_VERSION}.tar.gz -O - | tar xz
    cd chapel-${CHPL_VERSION}
    make -j ${SLURM_CPUS_PER_TASK}
    cd ..
fi


CHPL_BIN_SUBDIR=`"$CHPL_HOME"/util/chplenv/chpl_bin_subdir.py`
export PATH="$PATH":"$CHPL_HOME/bin/$CHPL_BIN_SUBDIR:$CHPL_HOME/util"

# Start computation
export GASNET_SSH_SERVERS=`scontrol show hostnames | xargs echo`

# Number of task per local = ${SLURM_CPUS_PER_TASK}
# Just to test
sed -i "s/tasksPerLocale = 1/tasksPerLocale = 128/" $CHPL_HOME/examples/hello6-taskpar-dist.chpl
chpl -o hello $CHPL_HOME/examples/hello6-taskpar-dist.chpl

srun ./hello -nl ${SLURM_NNODES}




#NUM_TRIALS=${1:-1}
#echo -e \#\#\# Running hello6 for $NUM_TRIALS trials \#\#\#
#loc=${SLURM_NNODES}
## update .numlocales and .good files for current number of nodes
#echo $loc > $CHPL_HOME/examples/hello6-taskpar-dist.numlocales
#rm -f ~/hello6-taskpar-dist.good
#for ((i = 0 ; i < $loc ; i++ )); do
#  echo "Hello, world! (from locale $i of $loc)" >> ~/hello6-taskpar-dist.good
#done
#LC_ALL="" LC_COLLATE="C" LANG="en_US.UTF-8" sort ~/hello6-taskpar-dist.good > $CHPL_HOME/examples/hello6-taskpar-dist.good
#rm -f ~/hello6-taskpar-dist.good
#start_test $CHPL_HOME/examples/hello6-taskpar-dist.chpl --numtrials $NUM_TRIALS




