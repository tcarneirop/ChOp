
#!/bin/sh

# oarsub -q production -p "cluster='grvingt'" -l nodes=32 -t allow_classic_ssh -I

# setup env for Chapel 1.23 using ofi
setupChplenv() {

  export LC_CTYPE=en_US.UTF-8
  export LC_ALL=en_US.UTF-8

  export GASNET_SSH_SERVERS=$(uniq $OAR_NODEFILE | tr '\n' ' ')
  module load gcc/6.4.0_gcc-6.4.0
  module load libfabric
  export CHPL_GASNET_MORE_CFG_OPTIONS="--with-ofi-spawner=ssh --disable-mpi-compat"
  # Ignore our errors about ofi/psm not being supported
  export CHPL_GASNET_ALLOW_BAD_SUBSTRATE=true

  export CHPL_HOME=~/chapel-1.23.0
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

  NUM_T_LOCALE=$(cat /proc/cpuinfo | grep processor | wc -l)
  export CHPL_RT_NUM_THREADS_PER_LOCALE=$NUM_T_LOCALE

  #export CHPL_RT_NUM_THREADS_PER_LOCALE=MAX_LOGICAL
  # no idea if this is needed
  export HFI_NO_CPUAFFINITY=1

  # Use ssh spawning -- I couldn't get mpi spawner working
  export GASNET_OFI_SPAWNER='ssh'
}

downloadChpl() {
  setupChplenv

  if [ ! -d "$CHPL_HOME" ]; then
    cd ~/
    # Download Chapel 1.23
    #wget https://github.com/chapel-lang/chapel/releases/download/1.22.0/chapel-1.22.0.tar.gz
    #tar xzf chapel-1.23.0.tar.gz
	# Replace GASNet-EX with GASNet-1
    wget https://gasnet.lbl.gov/download/GASNet-1.32.0.tar.gz
    tar xzf GASNet-1.32.0.tar.gz
    rm -rf $CHPL_HOME/third-party/gasnet/gasnet-src
    mv GASNet-1.32.0 $CHPL_HOME/third-party/gasnet/gasnet-src
    wget https://raw.githubusercontent.com/chapel-lang/chapel/release/1.19/runtime/src/launch/gasnetrun_common/gasnetrun_common.h
    mv gasnetrun_common.h $CHPL_HOME/runtime/src/launch/gasnetrun_common/gasnetrun_common.h
    setupChplenv
    
  fi

}

buildChpl() {
  setupChplenv
  downloadChpl
  cd $CHPL_HOME
  make -j `nproc`
  make test-venv
}

# run hello6 $NUM_TRIALS times using our start_test infrastructure
testChplHello() {
 
  NUM_TRIALS=${1:-1}
  echo -e \#\#\# Running hello6 for $NUM_TRIALS trials \#\#\#
  loc=$(uniq $OAR_NODEFILE | wc -l)
  #let loc = 2  
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





buildChpl
testChplHello
#callTestsHello
