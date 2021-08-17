#!/bin/bash

# Use gcc as a target compiler (faster compilation/execution)
module unload PrgEnv-cray
module load PrgEnv-gnu
module unload perftools-base

# Load hugepages (required for good comm=ugni perf)
module load craype-hugepages16M

# Load Chapel (only if you want to use 1.19 instead of building from source)
module load chapel

echo -e "### modules -- done ###"
# Use lustre
export LUS_HOME=/lus/scratch/$USER
cd ${LUS_HOME}

echo -e "### LUS HOME exported ###"

# Wrapper to reserve broadwell nodes for an hour. Just call `use_bw` to reserve
# 16 nodes, or use `use_bw $num_nodes` for however many nodes you want
use_bw() {
  num_nodes=${1:-32}
  export CHPL_LAUNCHER_CORES_PER_LOCALE=88
  qsub -V -N "chpl-exp" -l walltime=00:10:00 -l place=scatter,select=$num_nodes -I
}



echo -e "### use bw done ###"
#```
#That just loads some appropriate modules and adds a wrapper to reserve nodes. If you put that in your bashrc you should be able to test with something like:

#```

echo "coforall loc in Locales do on loc do writeln((here.name, here.maxTaskPar));" > f.chpl
chpl f.chpl
use_bw
./f -nl 16
# should print out 16 node names and 44 cores each.
#```

#You can also do a quick sanity check to make sure performance looks good. Something like:

#```
chpl $CHPL_HOME/examples/benchmarks/hpcc/stream.chpl --fast
#use_bw
./stream -nl 16
# should be ~1530 GB/s
#```
