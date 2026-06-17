#!/usr/bin/env bash


setupChplenv() {

  echo -e \#\#\# Building Chapel runtime 2.80 AMD GPU - $(rocm_agent_enumerator | grep -v gfx000 | sort -u | head -1) - MULTI Locale - ibv \#\#\#
  echo -e " ### loading modules...  ### "

ml cmake/3.23.3_gcc-10.4.0
ml gcc/13.2.0_gcc-10.4.0
ml python/3.10.8_gcc-10.4.0 

sudo-g5k apt-get update
sudo-g5k apt-get install -y hipcc
sudo-g5k apt-get install -y rocprim-dev hipcub-dev

export LD_LIBRARY_PATH=$LLVM_ROOT/lib:/opt/rocm-6.3.3/lib:$LD_LIBRARY_PATH

  export HIP_PATH=/opt/rocm-6.3.3/
  export ROCM_PATH=/opt/rocm-6.3.3/
  export DEVICE_LIB_PATH=/opt/rocm-6.3.3/amdgcn/bitcode/
  export CHPL_HOME=~/chapel-2.8.0
  export CHPL_LLVM=bundled
  export CHPL_LOCALE_MODEL=gpu
  export CHPL_HOST_PLATFORM=`$CHPL_HOME/util/chplenv/chpl_platform.py`
  export CHPL_GPU=amd
  export CHOP_HOME=~/ChOp
  export CHPL_ROCM_PATH=/opt/rocm-6.3.3/
  export CHPL_GPU_ARCH=$(rocm_agent_enumerator | grep -v gfx000 | sort -u | head -1)
  export CHPL_GPU_MEM_STRATEGY=array_on_device


  #export CHPL_RT_NUM_GPUS_PER_LOCAL=$(rocm_agent_enumerator | wc -l )

  export CHPL_COMM='gasnet'
  export CHPL_COMM_SUBSTRATE='ibv'
  export CHPL_TARGET_CPU='native'
  export GASNET_QUIET=1

  export GASNET_IBV_SPAWNER=ssh
  export GASNET_PHYSMEM_MAX='0.667'

  export CHOP_HOME=~/ChOp
  export LD_LIBRARY_PATH="$LD_LIBRARY_PATH":"$CHOP_HOME"/libs

  # no idea if this is needed
  export HFI_NO_CPUAFFINITY=1
  export LD_LIBRARY_PATH=$LLVM_ROOT/lib:/opt/rocm-6.3.3/lib:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH="$LD_LIBRARY_PATH":"$CHOP_HOME"/libs
  CHPL_BIN_SUBDIR=`"$CHPL_HOME"/util/chplenv/chpl_bin_subdir.py`
  export PATH="$PATH":"$CHPL_HOME/bin/$CHPL_BIN_SUBDIR:$CHPL_HOME/util"

  echo $LD_LIBRARY_PATH
  export MANPATH="$MANPATH":"$CHPL_HOME"/man


  NUM_T_LOCALE=$(cat /proc/cpuinfo | grep processor | wc -l)

  export CHPL_RT_NUM_THREADS_PER_LOCALE=$NUM_T_LOCALE

}


buildChpl() {
  setupChplenv
  pushd $CHPL_HOME
  nice make -j `nproc`
  make test-venv
  popd
  echo -e \#\#\# DONE \#\#\#
  echo -e \#\#\# Chapel runtime 2.80 AMD GPU - $(rocm_agent_enumerator | grep -v gfx000 | sort -u | head -1) - MULTI Locale - ibv  \#\#\#

}

setSSHServers() {
  export GASNET_SSH_SERVERS=$(uniq $OAR_NODEFILE | tr '\n' ' ')
}

setSSHServers
buildChpl
