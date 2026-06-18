#ifndef WARMUP_HPP
#define WARMUP_HPP



// A tiny dummy kernel to force physical wave dispatch and queue wake-up
__global__ void dummy_warmup_kernel(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

void gpu_warmup_improved(int device_id) {
    float* d_warmup = nullptr;
    size_t allocation_size = 1024 * 1024 * sizeof(float); // 4MB to force actual MMU mapping

    // Force real physical memory allocation
    if (hipMalloc(&d_warmup, allocation_size) != hipSuccess) return;
    if (hipMemset(d_warmup, 0, allocation_size) != hipSuccess) return;

    // Wake up execution queues via actual kernel launch
    hipLaunchKernelGGL(dummy_warmup_kernel, dim3(1), dim3(64), 0, 0, d_warmup);
    
    hipDeviceSynchronize();
   
    hipFree(d_warmup);
}

void warmup_all_gpus(const int repetitions) {
    int num_devices = 0;
    if (hipGetDeviceCount(&num_devices) != hipSuccess || num_devices == 0) {
        return;
    }

    printf("\nFound %d HIP device(s). Initializing full context and P2P matrix...\n", num_devices);

    // Phase 1: Establish P2P connections explicitly to prevent runtime lag during the main execution
    for (int i = 0; i < num_devices; ++i) {
        hipSetDevice(i);
        for (int j = 0; j < num_devices; ++j) {
            if (i != j) {
                int can_access = 0;
                hipDeviceCanAccessPeer(&can_access, i, j);
                if (can_access) {
                    // This forces the driver to map memory spaces across Infinity Fabric/PCIe instantly
                    hipDeviceEnablePeerAccess(j, 0);
                }
            }
        }
    }

    // Phase 2: Warm up hardware queues across all devices sequentially
    for (int dev = 0; dev < num_devices; ++dev) {
        printf("Warming up GPU %d\n", dev);
        hipSetDevice(dev);

        for (int i = 0; i < repetitions; ++i) {
            gpu_warmup_improved(dev);
        }
    }

    // Restore primary device context state
    if (num_devices > 0) {
        hipSetDevice(0);
    }
}


#endif
