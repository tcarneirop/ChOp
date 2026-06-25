#ifndef HIP_HEADERS_HPP
#define HIP_HEADERS_HPP

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(hipError_t code, const char *file, int line, bool abort=true)
{
	if (code != hipSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


void HIP_call_queens(int size, int initial_depth, unsigned long long n_explorers, 
    QueenRoot *__restrict__ root_prefixes_h ,
	unsigned long long *__restrict__ vector_of_tree_size_h, 
    unsigned long long *__restrict__ sols_h, int gpu_id, int block_size){
  
    unsigned long long *vector_of_tree_size_d;
    unsigned long long *sols_d;
    QueenRoot *root_prefixes_d;

    hipSetDevice(gpu_id);
    hipFree(0);

    int num_blocks = ceil((double)n_explorers/block_size);

    
    hipMalloc((void**) &root_prefixes_d,n_explorers*sizeof(QueenRoot));
    hipMalloc((void**) &vector_of_tree_size_d,n_explorers*sizeof(unsigned long long));    
    hipMalloc((void**) &sols_d,n_explorers*sizeof(unsigned long long));
    hipMemcpy(root_prefixes_d, root_prefixes_h, n_explorers * sizeof(QueenRoot), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(BP_queens_root_dfs, num_blocks, block_size, 0, 0, 
        size,n_explorers,initial_depth,root_prefixes_d, vector_of_tree_size_d,sols_d);


    hipMemcpy(vector_of_tree_size_h,vector_of_tree_size_d,n_explorers*sizeof(unsigned long long),hipMemcpyDeviceToHost);
    hipMemcpy(sols_h,sols_d,n_explorers*sizeof(unsigned long long),hipMemcpyDeviceToHost);
    
}


#endif