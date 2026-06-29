#ifndef CUDA_HEADERS_HPP
#define CUDA_HEADERS_HPP

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void CUDA_call_queens(int size, int initial_depth, unsigned long long n_explorers, 
    QueenRoot *__restrict__ root_prefixes_h ,
	unsigned long long *__restrict__ vector_of_tree_size_h, 
    unsigned long long *__restrict__ sols_h, int gpu_id, int block_size){
  
    unsigned long long *vector_of_tree_size_d;
    unsigned long long *sols_d;
    QueenRoot *root_prefixes_d;

    cudaSetDevice(gpu_id);
    cudaFree(0);

    int num_blocks = ceil((double)n_explorers/block_size);
    unsigned long long zero = 0;

    cudaMalloc((void**) &root_prefixes_d,n_explorers*sizeof(QueenRoot));
    cudaMalloc((void**) &vector_of_tree_size_d,sizeof(unsigned long long)); 
    
    cudaMemcpy(vector_of_tree_size_d, &zero, sizeof(unsigned long long), cudaMemcpyHostToDevice);
    
    cudaMalloc((void**) &sols_d,sizeof(unsigned long long));
    cudaMemcpy(sols_d,  &zero,  sizeof(unsigned long long), cudaMemcpyHostToDevice);
    
    cudaMemcpy(root_prefixes_d, root_prefixes_h, n_explorers * sizeof(QueenRoot), cudaMemcpyHostToDevice);

    CUDA_HIP__queens_dfs_enumeration<<< num_blocks,block_size>>>(size,n_explorers,initial_depth,root_prefixes_d, vector_of_tree_size_d,sols_d);

    cudaMemcpy(vector_of_tree_size_h,vector_of_tree_size_d,sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(sols_h,sols_d                              ,sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    
}


#endif