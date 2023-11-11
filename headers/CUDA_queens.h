#ifndef CUDA_QUEENS_H
#define CUDA_QUEENS_H

#define CUDA_QUEENS_BLOCK_SIZE_ 	128

#include "queens_node.h"

#ifdef __cplusplus
extern "C" {
#endif
void  CUDA_call_queens(short size, int initial_depth, unsigned int n_explorers, QueenRoot *root_prefixes_h, 
	unsigned long long *vector_of_tree_size_h, unsigned long long *sols_h, int gpu_id);
#ifdef __cplusplus
}
#endif



#endif
