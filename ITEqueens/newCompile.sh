
#!/bin/bash

echo " ### Cleaning... ###"

rm libs/*.so

echo " ### starting CUDA compilation ### "
nvcc --shared -o libs/libGPUAPI.so ${GPUITE_HOME}/src/GPUAPI.cu --compiler-options '-fPIC -O3' -I${CUDA_HOME}/include -L${CUDA_HOME}/lib -lcudart
nvcc --shared -o libs/libite.so kernels/GPU_queens_kernels.cu  --compiler-options '-fPIC -O3' -I${CUDA_HOME}/include -L${CUDA_HOME}/lib -lcudart
nvcc --shared -o libs/libutil.so kernels/GPU_aux.cu  --compiler-options '-fPIC -O3' -I${CUDA_HOME}/include -L${CUDA_HOME}/lib -lcudart

echo " ### end of CUDA compilation ### "

echo " ### starting compilation ### "
# -s debugDistributedIters=true 
#chpl -L. -ladd -M modules --fast -s timeDistributedIters=true implementation/fsp_gen.c implementation/simple_bound.c implementation/johnson_bound.c implementation/aux.c main.chpl -o  fsp.out
chpl -Llibs -lite -lutil -lGPUAPI -M modules -M ${GPUITE_HOME}/src --fast -s timeDistributedIters=true -s infoDistributedIters=true -s debugGPUIterator=true ${GPUITE_HOME}/src/GPUAPI.h newMain.chpl -o  newQueens.out

#source export.sh 

echo " ### end of compilation ### "

# -suseBulkTransfer=false -sdebugBlockDistBulkTransfer=true -sdebugBulkTransfer=true
