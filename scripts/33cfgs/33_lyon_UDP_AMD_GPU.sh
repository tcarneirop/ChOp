#!/usr/bin/env bash

# command to reserve nodes:
#   oarsub -q production -p "cluster='grvingt'" -l nodes=32 -t allow_classic_ssh -I

# setup env for Chapel 1.24 using ofi
setupChplenv() {

 module use /grid5000/spack/share/spack/modules/linux-debian9-x86_64/
  #module load gcc/6.4.0_gcc-6.4.0
  module load cmake
  module load libfabric

  # Ignore our errors about ofi/psm not being supported
  #export CHPL_GASNET_ALLOW_BAD_SUBSTRATE=true

  export CHPL_HOME=~/chapel-1.33.0
 # if [ -d "$CHPL_HOME" ]; then
    CHPL_BIN_SUBDIR=`"$CHPL_HOME"/util/chplenv/chpl_bin_subdir.py`
    export PATH="$PATH":"$CHPL_HOME/bin/$CHPL_BIN_SUBDIR:$CHPL_HOME/util"
 #fi

  export MANPATH="$MANPATH":"$CHPL_HOME"/man
  export CHPL_LLVM=bundled
  #export FI_PROVIDER=psm2

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
  export CHPL_GPU=amd
  export CHPL_ROCM_PATH=/opt/rocm-4.5.0
  export CHPL_GPU_ARCH=gfx906

  export CHOP_HOME=~/ChOp
  export LD_LIBRARY_PATH="$LD_LIBRARY_PATH":"$CHOP_HOME"/libs

  # no idea if this is needed
  export HFI_NO_CPUAFFINITY=1

  # Use ssh spawning (and avoid mpi) -- I couldn't get mpi spawner working
  #export CHPL_GASNET_MORE_CFG_OPTIONS="--with-ofi-spawner=ssh --disable-mpi-compat"
  # TODO force psm provider
  #export GASNET_OFI_SPAWNER='ssh'

  NUM_T_LOCALE=$(cat /proc/cpuinfo | grep processor | wc -l)

  export CHPL_RT_NUM_THREADS_PER_LOCALE=$NUM_T_LOCALE




}

downloadChpl() {
  setupChplenv
  if [ ! -d "$CHPL_HOME" ]; then
    cd ~/
    # Download Chapel 1.27
    wget -c https://github.com/chapel-lang/chapel/releases/download/1.33.0/chapel-1.33.0.tar.gz -O - | tar xz
    setupChplenv
  fi
}

buildChpl() {
  setupChplenv
  downloadChpl
  pushd $CHPL_HOME
  nice make -j `nproc`
  make test-venv
  popd
}

setSSHServers() {
  export GASNET_SSH_SERVERS=$(uniq $OAR_NODEFILE | tr '\n' ' ')
}

setSSHServers
buildChpl
