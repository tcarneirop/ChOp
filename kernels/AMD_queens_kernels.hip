#include <hip/hip_runtime.h>
#include <stdio.h>
#include "../headers/AMD_queens.h"
//#include "../headers/queens_node.h"


__device__  inline bool AMD_queens_stillLegal(const char *__restrict__ board, const int r){

  bool safe = true;
  int i, rev_i, offset;
  const char base = board[r];
  // Check vertical
  for ( i = 0, rev_i = r-1, offset=1; i < r; ++i, --rev_i, offset++)
    safe &= !((board[i] == base) | ( (board[rev_i] == base-offset) |
                                     (board[rev_i] == base+offset)));
  return safe;
}



__global__ void AMD_queens_root_dfs( const int N, const unsigned int nPrefixes, 
    const int initial_depth,
    QueenRoot *__restrict__ root_prefixes,
    unsigned long long int *__restrict__ vector_of_tree_size, 
    unsigned long long int *__restrict__ sols){

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
            }else if (!(flag &  mask ) && AMD_queens_stillLegal(board, depth)){

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



extern "C" void AMD_call_queens(short size, int initial_depth, unsigned int n_explorers, QueenRoot *root_prefixes_h ,
	unsigned long long *vector_of_tree_size_h, unsigned long long *sols_h, int gpu_id){
    
    hipSetDevice(gpu_id);
    hipFree(0);
   
    unsigned long long *vector_of_tree_size_d;
    unsigned long long *sols_d;
    QueenRoot *root_prefixes_d;

    int num_blocks = ceil((double)n_explorers/AMD_QUEENS_BLOCK_SIZE);

    hipMalloc((void**) &vector_of_tree_size_d,n_explorers*sizeof(unsigned long long));
    hipMalloc((void**) &sols_d,n_explorers*sizeof(unsigned long long));
    hipMalloc((void**) &root_prefixes_d,n_explorers*sizeof(QueenRoot));

    hipMemcpy(root_prefixes_d, root_prefixes_h, n_explorers * sizeof(QueenRoot),  hipMemcpyHostToDevice);
    

    hipLaunchKernelGGL(AMD_queens_root_dfs, num_blocks, AMD_QUEENS_BLOCK_SIZE, 0, 0, 
        size,
        n_explorers,
        initial_depth,
        root_prefixes_d,
        vector_of_tree_size_d,
        sols_d
        );

    //gpuErrchk( cudaPeekAtLastError() );
    //gpuErrchk( cudaDeviceSynchronize() );

    hipMemcpy(vector_of_tree_size_h,vector_of_tree_size_d,n_explorers*sizeof(unsigned long long), hipMemcpyDeviceToHost);
    hipMemcpy(sols_h,sols_d,n_explorers*sizeof(unsigned long long), hipMemcpyDeviceToHost);
    
    hipFree(vector_of_tree_size_d);
    hipFree(sols_d);
    hipFree(root_prefixes_d);
}
