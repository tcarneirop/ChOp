#ifndef QUEENS_OMP_AUX
#define QUEENS_OMP_AUX



void  get_load_each_gpu(unsigned long long gpu_load, int num_gpus, unsigned long long *device_load){

    for(int device = 0; device<num_gpus;++device){
        device_load[device] = gpu_load/num_gpus;
        if(device == (num_gpus-1)){
            device_load[device]+= gpu_load%num_gpus;
        }
    }
}//////



#endif