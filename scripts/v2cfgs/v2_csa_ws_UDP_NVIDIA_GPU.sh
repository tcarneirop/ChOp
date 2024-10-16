#!/usr/bin/env bash

setupChplenv() {


  ml CMake
echo "ml cmake"

  module load libfabric
echo "ml lib"

  ml CUDA/12.2.0

echo "ml cuda"
ml Python

  export CHPL_HOME="/scratch/carnei26/chapel-2.2.0"
 # if [ -d "$CHPL_HOME" ]; then
    CHPL_BIN_SUBDIR=`"$CHPL_HOME"/util/chplenv/chpl_bin_subdir.py`
    export PATH="$PATH":"$CHPL_HOME/bin/$CHPL_BIN_SUBDIR:$CHPL_HOME/util"
 #fi

  export MANPATH="$MANPATH":"$CHPL_HOME"/man
  export CHPL_LLVM=bundled

  # use gasnet-ofi -- from the psm docs "Users of Intel(R) Omni-Path Fabric are
  # recommended to use ofi-conduit"
  export CHPL_COMM='gasnet'
  export CHPL_COMM_SUBSTRATE='udp'
  export CHPL_TARGET_CPU='native'
  export GASNET_QUIET=1

  # Use SSH to spawn jobs
  export GASNET_SPAWNFN=S
  # Which ssh command should be used? ssh is the default.
  export GASNET_SSH_CMD=ssh
  # Disable X11 forwarding
  export GASNET_SSH_OPTIONS=-x

  #export GASNET_IBV_SPAWNER=ssh
  export GASNET_PHYSMEM_MAX='0.667'

  export CHPL_GPU_MEM_STRATEGY=array_on_device
  export CHPL_LOCALE_MODEL=gpu
  export CHPL_GPU=nvidia
  export CHPL_CUDA_PATH="/shared/csa/software/CUDA/12.2.0/"

  export CHOP_HOME=~/ChOp
  export LD_LIBRARY_PATH="$LD_LIBRARY_PATH":"$CHOP_HOME"/libs


  NUM_T_LOCALE=$(cat /proc/cpuinfo | grep processor | wc -l)

  export CHPL_RT_NUM_THREADS_PER_LOCALE=$NUM_T_LOCALE


}

buildChpl() {
  setupChplenv
  pushd $CHPL_HOME
  nice make -j `nproc`
  make test-venv
  popd
}

setSSHServers() {
  export GASNET_SSH_SERVERS=$(uniq $PBS_NODEFILE | tr '\n' ' ')
}



echo "primeiro"
setSSHServers
echo "segundo"
buildChpl
