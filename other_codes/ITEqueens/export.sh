
#!/bin/bash

echo " ### exporting...  ### "

#export GPUITE_HOME=$(pwd)/chapel-gpu
#export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/libs

echo $LD_LIBRARY_PATH






