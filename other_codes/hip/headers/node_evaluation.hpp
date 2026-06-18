#ifndef HELPER_HPP
#define HELPER_HPP


#define _EMPTY_      -1
#define MAX_SIZE     24



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

double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}


typedef struct queen_root{
		unsigned int control;
		int8_t board[12]; //maximum depth of the solution space.
} QueenRoot;


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(hipError_t code, const char *file, int line, bool abort=true)
{
	if (code != hipSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

inline void prefixesHandleSol(QueenRoot *root_prefixes, unsigned int flag,
															const char *board, int initialDepth, int num_sol)
{
	root_prefixes[num_sol].control = flag;
	for(int i = 0; i<initialDepth;++i)
		root_prefixes[num_sol].board[i] = board[i];
}


__device__ __host__ inline bool GPU_queens_stillLegal(const char *__restrict__  board, const int r){

	bool safe = true;
	int i, rev_i, offset;
	const char base = board[r];
	// Check vertical
	for ( i = 0, rev_i = r-1, offset=1; i < r; ++i, --rev_i, offset++)
		safe &= !((board[i] == base) | ( (board[rev_i] == base-offset) |(board[rev_i] == base+offset)));
	return safe;
}




#endif
