
#!/bin/bash

echo " ### Cleaning... ###"
rm *.so
rm main
echo " ### starting CUDA compillation ### "
nvcc --shared -o libadd.so vector_add.cu --compiler-options '-fPIC' -I/usr/local/cuda/include -L/usr/local/cuda/lib -lcudart
echo " ### end of CUDA compillation ### "

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)
echo $LD_LIBRARY_PATH

echo "### starting CHPL compillation ###"
chpl -L. -ladd main.chpl





