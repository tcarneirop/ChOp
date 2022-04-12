#include <cuda.h>
#include <stdio.h>

#include "../headers/GPU_aux.h"


extern "C" int GPU_device_count(){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    return deviceCount;
}

extern "C" int GPU_set_device(int device){
    return cudaSetDevice(device);
}
