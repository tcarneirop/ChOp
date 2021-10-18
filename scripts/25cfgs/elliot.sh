#!/usr/bin/env bash

# command to reserve nodes:
#   oarsub -q production -p "cluster='grvingt'" -l nodes=32 -t allow_classic_ssh -I

# setup env for Chapel 1.24 using ofi
setupChplenv() {
  module load gcc/6.4.0_gcc-6.4.0
  module load libfabric

  # Ignore our errors about ofi/psm not being supported
  export CHPL_GASNET_ALLOW_BAD_SUBSTRATE=true

  export CHPL_HOME=~/chapel-1.25.0
  if [ -d "$CHPL_HOME" ]; then
    CHPL_BIN_SUBDIR=`"$CHPL_HOME"/util/chplenv/chpl_bin_subdir.py`
    export PATH="$PATH":"$CHPL_HOME/bin/$CHPL_BIN_SUBDIR:$CHPL_HOME/util"
  fi

  # use gasnet-ofi -- from the psm docs "Users of Intel(R) Omni-Path Fabric are
  # recommended to use ofi-conduit"
  export CHPL_COMM='gasnet'
  export CHPL_COMM_SUBSTRATE='ofi'
  export CHPL_TARGET_CPU='native'
  export GASNET_QUIET=1

  # no idea if this is needed
  export HFI_NO_CPUAFFINITY=1

  # Use ssh spawning (and avoid mpi) -- I couldn't get mpi spawner working
  export CHPL_GASNET_MORE_CFG_OPTIONS="--with-ofi-spawner=ssh --disable-mpi-compat"
  # TODO force psm provider
  export GASNET_OFI_SPAWNER='ssh'
}

downloadChpl() {
  setupChplenv

  if [ ! -d "$CHPL_HOME" ]; then
    cd ~/
    # Download Chapel 1.24
    wget -c https://github.com/chapel-lang/chapel/releases/download/1.24.0/chapel-1.24.0.tar.gz -O - | tar xz

    # Replace GASNet-EX with GASNet-1
    wget -c https://gasnet.lbl.gov/download/GASNet-1.32.0.tar.gz -O - | tar xz
    rm -rf $CHPL_HOME/third-party/gasnet/gasnet-src
    mv GASNet-1.32.0 $CHPL_HOME/third-party/gasnet/gasnet-src

    # Replace ofi launcher with older one (current version throws `-c 0`, which
    # isn't supported by older gasnets
    wget https://raw.githubusercontent.com/chapel-lang/chapel/release/1.19/runtime/src/launch/gasnetrun_common/gasnetrun_common.h
    mv gasnetrun_common.h $CHPL_HOME/runtime/src/launch/gasnetrun_common/gasnetrun_common.h

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

# run hello6 $NUM_TRIALS times using our start_test infrastructure
testChplHello() {
  export GASNET_SSH_SERVERS=$(uniq $OAR_NODEFILE | tr '\n' ' ')

  NUM_TRIALS=${1:-1}
  echo -e \#\#\# Running hello6 for $NUM_TRIALS trials \#\#\#
  loc=$(uniq $OAR_NODEFILE | wc -l)
  # update .numlocales and .good files for current number of nodes
  echo $loc > $CHPL_HOME/examples/hello6-taskpar-dist.numlocales
  rm -f ~/hello6-taskpar-dist.good
  for ((i = 0 ; i < $loc ; i++ )); do
    echo "Hello, world! (from locale $i of $loc)" >> ~/hello6-taskpar-dist.good
  done
  LC_ALL="" LC_COLLATE="C" LANG="en_US.UTF-8" sort ~/hello6-taskpar-dist.good > $CHPL_HOME/examples/hello6-taskpar-dist.good
  rm -f ~/hello6-taskpar-dist.good

  start_test $CHPL_HOME/examples/hello6-taskpar-dist.chpl --numtrials $NUM_TRIALS
}

setSSHServers() {
  export GASNET_SSH_SERVERS=$(uniq $OAR_NODEFILE | tr '\n' ' ')
}

reserveNodes() {
  num_nodes=${1:-2}
  oarsub -q production -p "cluster='grvingt'" -l nodes=$num_nodes -t allow_classic_ssh -I
}

setupChplenv