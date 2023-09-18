#include <cuda.h>
#include <stdio.h>

#include "../headers/GPU_queens.h"
#define _QUEENS_BLOCK_SIZE_ 	128
#define _EMPTY_ -1

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__  inline bool GPU_queens_stillLegal(const char *board, const int r){

  bool safe = true;
  int i, rev_i, offset;
  const char base = board[r];
  // Check vertical
  for ( i = 0, rev_i = r-1, offset=1; i < r; ++i, --rev_i, offset++)
    safe &= !((board[i] == base) | ( (board[rev_i] == base-offset) |
                                     (board[rev_i] == base+offset)));
  return safe;
}



__global__ void BP_queens_root_dfs( const int N, const unsigned int nPrefixes, 
    const int initial_depth,
    QueenRoot *root_prefixes,
    unsigned long long int * vector_of_tree_size, 
    unsigned long long int *sols){

       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       
    if (idx < nPrefixes) {
        unsigned int flag = 0;
        char board[32];
        int N_l = N;
        int i, depth;
        unsigned long long  qtd_sols_thread = 0ULL;
        int depthGlobal = initial_depth;
        unsigned long long int tree_size = 0ULL;

        for (i = 0; i < N_l; ++i) {
            board[i] = _EMPTY_;
        }

        flag = root_prefixes[idx].control;

        for (i = 0; i < depthGlobal; ++i)
            board[i] = root_prefixes[idx].board[i];

        depth=depthGlobal;

        do{

            board[depth]++;
            const int mask = 1<<board[depth];

            if(board[depth] == N_l){
                board[depth] = _EMPTY_;
                depth--;
                flag &= ~(1<<board[depth]);
            }else if (!(flag &  mask ) && GPU_queens_stillLegal(board, depth)){

                    ++tree_size;
                    flag |= mask;

                    depth++;

                    if (depth == N_l) { //sol
                        ++qtd_sols_thread ;

                        depth--;
                        flag &= ~mask;
                    }
                }
            }while(depth >= depthGlobal); //FIM DO DFS_BNB

        sols[idx] = qtd_sols_thread ;
        vector_of_tree_size[idx] = tree_size;
    }//if
}//kernel
////////



extern "C" void GPU_call_cuda_queens(short size, int initial_depth, unsigned int n_explorers, QueenRoot *root_prefixes_h ,
	unsigned long long *vector_of_tree_size_h, unsigned long long *sols_h, int gpu_id){
    
    cudaSetDevice(gpu_id);
    //cudaFree(0);
   // cudaFuncSetCacheConfig(BP_queens_root_dfs,cudaFuncCachePreferL1);

    unsigned long long *vector_of_tree_size_d;
    unsigned long long *sols_d;
    QueenRoot *root_prefixes_d;

    int num_blocks = ceil((double)n_explorers/_QUEENS_BLOCK_SIZE_);

    cudaMalloc((void**) &vector_of_tree_size_d,n_explorers*sizeof(unsigned long long));
    cudaMalloc((void**) &sols_d,n_explorers*sizeof(unsigned long long));
    cudaMalloc((void**) &root_prefixes_d,n_explorers*sizeof(QueenRoot));

    cudaMemcpy(root_prefixes_d, root_prefixes_h, n_explorers * sizeof(QueenRoot), cudaMemcpyHostToDevice);
    
    BP_queens_root_dfs<<< num_blocks,_QUEENS_BLOCK_SIZE_>>> (size,n_explorers,initial_depth,root_prefixes_d, vector_of_tree_size_d,sols_d);
    //gpuErrchk( cudaPeekAtLastError() );
    //gpuErrchk( cudaDeviceSynchronize() );

    cudaMemcpy(vector_of_tree_size_h,vector_of_tree_size_d,n_explorers*sizeof(unsigned long long),cudaMemcpyDeviceToHost);
    cudaMemcpy(sols_h,sols_d,n_explorers*sizeof(unsigned long long),cudaMemcpyDeviceToHost);
    
    cudaFree(vector_of_tree_size_d);
    cudaFree(sols_d);
    cudaFree(root_prefixes_d);
}
