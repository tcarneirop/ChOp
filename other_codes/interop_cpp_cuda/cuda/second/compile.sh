
#!/bin/bash

echo " ### Cleaning... ###"

rm libadd.so 
rm main
rm main_real

echo " ### starting CUDA compillation ### "
nvcc --shared -o libadd.so kernels/vector_add.cu --compiler-options '-fPIC' -I/usr/local/cuda/include -L/usr/local/cuda/lib -lcudart
echo " ### end of CUDA compillation ### "


echo "### starting CHPL compillation ###"
chpl -L. -ladd main.chpl





