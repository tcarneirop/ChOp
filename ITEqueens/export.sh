
#!/bin/bash

echo " ### exporting...  ### "
cd ..

export GPUITE_HOME=$(pwd)/chapel-gpu
export CUDA_HOME=/usr/lib/cuda

cd ITEqueens

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/libs

echo $LD_LIBRARY_PATH






