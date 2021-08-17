
#!/bin/bash

echo " ### Cleaning... ###"

rm libs/libadd.so
rm bin/chop.out
rm bin/chop.out_real

echo " ### starting CUDA compilation ### "

nvcc --shared -o libs/libadd.so kernels/GPU_queens_kernels.cu  --compiler-options '-fPIC -O3' -I/usr/local/cuda/include -L/usr/local/cuda/lib -lcudart
nvcc --shared -o libs/libutil.so kernels/GPU_aux.cu  --compiler-options '-fPIC -O3' -I/usr/local/cuda/include -L/usr/local/cuda/lib -lcudart

echo " ### end of CUDA compilation ### "

echo " ### starting compilation ### "
# -s debugDistributedIters=true 
#chpl -L. -ladd -M modules --fast -s timeDistributedIters=true implementation/fsp_gen.c implementation/simple_bound.c implementation/johnson_bound.c implementation/aux.c main.chpl -o  fsp.out
chpl -Llibs -ladd -lutil -M modules --fast -s queens_checkPointer=false -s timeDistributedIters=true -s infoDistributedIters=true implementation/fsp_gen.c implementation/simple_bound.c implementation/johnson_bound.c implementation/aux.c main.chpl -o  chop.out

#source export.sh

echo " ### end of compilation ### "
mv chop.out chop.out_real bin/
echo " ### copy is done ### "

ncompi=$(cat ncompilations)

var=$(($ncompi+1))


echo "### Number of compilations: " $var "####"
echo $var > ncompilations 


# -suseBulkTransfer=false -sdebugBlockDistBulkTransfer=true -sdebugBulkTransfer=true
