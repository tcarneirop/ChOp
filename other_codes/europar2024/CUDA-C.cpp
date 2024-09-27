module queens_GPU_call_device_search{require "headers/CUDA_queens.h";
	extern proc  CUDA_call_queens(size: uint(16), initial_depth:c_int, n_explorers:c_uint,root_prefixes_h: c_ptr(queens_node),vector_of_tree_size_h: c_ptr(c_ulonglong),sols_h: c_ptr(c_ulonglong),gpu_id:c_int): void;
	proc queens_GPU_call_device_search(const num_gpus: c_int, const size: uint(16), const depth: c_int,ref local_active_set: [] queens_node, const initial_num_prefixes: uint(64),const CPUP: real, const chunk: int ): (uint(64), uint(64)){
		var vector_of_tree_size_h: [0..#initial_num_prefixes] c_ulonglong;
		var sols_h: [0..#initial_num_prefixes] c_ulonglong;
		var cpu_load: c_uint = (CPUP * initial_num_prefixes):c_uint;
		var metrics: (uint(64),uint(64)) = (0:uint(64),0:uint(64));//
			coforall gpu_id in 0..#num_gpus:c_int do{
					////WE DO NOT NEED THIS FUNCTION AS IT IS ALSO USED IN CHAPEL-GPU
					var gpu_load: c_uint = GPU_mlocale_get_gpu_load(new_num_prefixes:c_uint, gpu_id:c_int, num_gpus);
					////WE need to add this function here
					var starting_position: c_uint = GPU_mlocale_get_starting_point(new_num_prefixes:c_uint,gpu_id:c_uint, num_gpus:c_uint, cpu_load:c_uint);
					var sol_ptr : c_ptr(c_ulonglong) = c_ptrTo(sols_h) + starting_position;
					var tree_ptr : c_ptr(c_ulonglong) = c_ptrTo(vector_of_tree_size_h) + starting_position;
					var nodes_ptr : c_ptr(queens_node) = c_ptrTo(local_active_set) + starting_position;
					CUDA_call_queens(size, depth, gpu_load:c_uint,nodes_ptr, tree_ptr, sol_ptr, gpu_id:c_int);
		var redTree = (+ reduce vector_of_tree_size_h):uint(64);
		var redSol  = (+ reduce sols_h):uint(64);
		return ((redSol,redTree)+metrics);}///

=17

/////////// offloading the search to the device


proc GPU_mlocale_get_starting_point(const survivors: c_uint, const gpu_id:c_uint,const num_gpus: c_uint, const cpu_load: c_uint ): c_uint{
		var cpu_end: c_uint = if cpu_load>0 then (cpu_load+1) else 0;
		return ((gpu_id*(survivors/num_gpus))+cpu_end);}////////////

=2

////////// pinters arithmetics 

#ifndef CUDA_QUEENS_H
#define CUDA_QUEENS_H
#define CUDA_QUEENS_BLOCK_SIZE_ 	128
#include "queens_node.h"
#ifdef __cplusplus
extern "C" {
#endif
void  CUDA_call_queens(short size, int initial_depth, unsigned int n_explorers, QueenRoot *root_prefixes_h, unsigned long long *vector_of_tree_size_h, unsigned long long *sols_h, int gpu_id);
#ifdef __cplusplus}
#endif 
#endif

=8

//// interoperability header

extern "C" void CUDA_call_queens(short size, int initial_depth, unsigned int n_explorers, QueenRoot *root_prefixes_h ,unsigned long long *vector_of_tree_size_h, unsigned long long *sols_h, int gpu_id){
    cudaSetDevice(gpu_id);
    int num_blocks = ceil((double)n_explorers/CUDA_QUEENS_BLOCK_SIZE_);
    cudaMalloc((void**) &vector_of_tree_size_d,n_explorers*sizeof(unsigned long long));
    cudaMalloc((void**) &sols_d,n_explorers*sizeof(unsigned long long));
    cudaMalloc((void**) &root_prefixes_d,n_explorers*sizeof(QueenRoot));
    cudaMemcpy(root_prefixes_d, root_prefixes_h, n_explorers * sizeof(QueenRoot), cudaMemcpyHostToDevice);
    BP_queens_root_dfs<<< num_blocks,CUDA_QUEENS_BLOCK_SIZE_>>> (size,n_explorers,initial_depth,root_prefixes_d, vector_of_tree_size_d,sols_d);
    cudaMemcpy(vector_of_tree_size_h,vector_of_tree_size_d,n_explorers*sizeof(unsigned long long),cudaMemcpyDeviceToHost);
    cudaMemcpy(sols_h,sols_d,n_explorers*sizeof(unsigned long long),cudaMemcpyDeviceToHost); 
    cudaFree(vector_of_tree_size_d);
    cudaFree(sols_d);
    cudaFree(root_prefixes_d);}

=9

/// C function for CUDA-C kernel calling --- H2D and D2H transfers 

__device__  inline bool GPU_queens_stillLegal(const char *__restrict__ board, const int r){
	bool safe = true;
  int i, rev_i, offset;
  const char base = board[r];
  for ( i = 0, rev_i = r-1, offset=1; i < r; ++i, --rev_i, offset++)
    safe &= !((board[i] == base) | ( (board[rev_i] == base-offset) | (board[rev_i] == base+offset)));
  return safe;}

 =5 

 /// subproblem evaluation -- true or false

__global__ void BP_queens_root_dfs( const int N, const unsigned int nPrefixes, const int initial_depth, QueenRoot *__restrict__ root_prefixes,unsigned long long int *__restrict__ vector_of_tree_size,  unsigned long long int *__restrict__ sols){
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nPrefixes) {
        unsigned int flag = 0;
    	char board[32];int N_l = N;int i, depth;
        unsigned long long  qtd_sols_thread = 0ULL;
        int depthGlobal = initial_depth;
        unsigned long long int tree_size = 0ULL;

        for (i = 0; i < N_l; ++i) board[i] = _EMPTY_;
        flag = root_prefixes[idx].control;

        for (i = 0; i < depthGlobal; ++i)  board[i] = root_prefixes[idx].board[i];

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

                    if (depth == N_l) { 

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

=24 
///CUDA-C kernel itself

CUDA_SRC_DIR := ./kernels
CUDA_PATH := $(CUDA_HOME)
CUDA_INCLUDE_DIR := $(CUDA_PATH)/include
CUDA_LIB_DIR := $(CUDA_PATH)/lib
LIBRARY_DIR := ./libs
chapelcuda: cuda dir
	chpl -s GPUCUDA=true -s GPUAMD=false -L$(LIBRARY_DIR) -lqueens -lutil -M $(CHPL_MODULES_DIR) --fast $(CHPL_DEBUG_FLAGS) main.chpl -o  $(BUILD_DIR)/chop.out
cuda: dir
	$(CUDA_PATH)/bin/nvcc --shared -o $(LIBRARY_DIR)/libqueens.so $(CUDA_SRC_DIR)/CUDA_queens_kernels.cu  --compiler-options '-fPIC -O3' -I$(CUDA_INCLUDE_DIR) -L$(CUDA_LIB_DIR) -lcudart
	$(CUDA_PATH)/bin/nvcc --shared -o $(LIBRARY_DIR)/libutil.so $(CUDA_SRC_DIR)/GPU_aux.cu  --compiler-options '-fPIC -O3' -I$(CUDA_INCLUDE_DIR) -L$(CUDA_LIB_DIR) -lcudart
dir:
	mkdir -p $(LIBRARY_DIR) mkdir -p $(BUILD_DIR)

=9

////// makefile
=====74 
