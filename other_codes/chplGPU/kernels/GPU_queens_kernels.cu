#include <cuda.h>
#include <stdio.h>

#include "../headers/GPU_queens.h"
#define _QUEENS_BLOCK_SIZE_ 128
#define _EMPTY_     -1
#define _MAX_ 32


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__device__  bool GPU_queens_stillLegal(const char *board, const int r){

  bool safe = true;
  int i;
  register int ld;
  register int rd;
  // Check vertical
  for ( i = 0; i < r; ++i)
    if (board[i] == board[r]) safe = false;
    // Check diagonals
    ld = board[r];  //left diagonal columns
    rd = board[r];  // right diagonal columns
    for ( i = r-1; i >= 0; --i) {
      --ld; ++rd;
      if (board[i] == ld || board[i] == rd) safe = false;
    }

    return safe;
}


__global__ void BP_queens_root_dfs(int N, unsigned int nPreFixos, int depthPreFixos,
    QueenRoot *root_prefixes,unsigned long long int *vector_of_tree_size, unsigned long long int *sols){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nPreFixos) {
        unsigned int flag = 0;
        unsigned int bit_test = 0;
        char board[32]; //representa o ciclo
        int N_l = N;
        int i, depth;
        unsigned long long  qtd_solucoes_thread = 0ULL;
        int depthGlobal = depthPreFixos;
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
            bit_test = 0;
            bit_test |= (1<<board[depth]);

            if(board[depth] == N_l){
                board[depth] = _EMPTY_;
                //if(block_ub > upper)   block_ub = upper;
            }else if (!(flag &  bit_test ) && GPU_queens_stillLegal(board, depth)){

                    ++tree_size;
                    flag |= (1ULL<<board[depth]);

                    depth++;

                    if (depth == N_l) { //sol
                        ++qtd_solucoes_thread;
                    }else continue;
                }else continue;

            depth--;
            flag &= ~(1ULL<<board[depth]);

            }while(depth >= depthGlobal); //FIM DO DFS_BNB

        sols[idx] = qtd_solucoes_thread;
        vector_of_tree_size[idx] = tree_size;
    }//if
}//kernel
////////

extern "C" void GPU_call_cuda_queens(short size, int initial_depth, unsigned int n_explorers, QueenRoot *root_prefixes_h ,
	unsigned long long *vector_of_tree_size_h, unsigned long long *sols_h, int gpu_id){


//@todo -- use the Chapel GPU API By Akihiro
    cudaSetDevice(gpu_id);
   // cudaFuncSetCacheConfig(BP_queens_root_dfs,cudaFuncCachePreferL1);


    unsigned long long *vector_of_tree_size_d;
    unsigned long long *sols_d;
    QueenRoot *root_prefixes_d;

    int num_blocks = ceil((double)n_explorers/_QUEENS_BLOCK_SIZE_);

    cudaMalloc((void**) &vector_of_tree_size_d,n_explorers*sizeof(unsigned long long));
    cudaMalloc((void**) &sols_d,n_explorers*sizeof(unsigned long long));
    cudaMalloc((void**) &root_prefixes_d,n_explorers*sizeof(QueenRoot));


    //@todo -- use the Chapel GPU API By Akihiro
    printf("%d  - %d  - %d  -\n", num_blocks, n_explorers, _QUEENS_BLOCK_SIZE_ );
    cudaMemcpy(root_prefixes_d, root_prefixes_h, n_explorers * sizeof(QueenRoot), cudaMemcpyHostToDevice);

    BP_queens_root_dfs<<< num_blocks,_QUEENS_BLOCK_SIZE_>>> (size,n_explorers,initial_depth,root_prefixes_d, vector_of_tree_size_d,sols_d);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    cudaMemcpy(vector_of_tree_size_h,vector_of_tree_size_d,n_explorers*sizeof(unsigned long long),cudaMemcpyDeviceToHost);
    cudaMemcpy(sols_h,sols_d,n_explorers*sizeof(unsigned long long),cudaMemcpyDeviceToHost);

    cudaFree(vector_of_tree_size_d);
    cudaFree(sols_d);
    cudaFree(root_prefixes_d);
    //After that, Chapel reduces the values
}
