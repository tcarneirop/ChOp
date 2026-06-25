#ifndef GPU_QUEENS_H
#define GPU_QUEENS_H

typedef struct queen_root{
	unsigned int control;
	int8_t board[12];
} QueenRoot;


#ifdef __cplusplus
extern "C" {
#endif
void GPU_call_cuda_queens(short size, int initial_depth, unsigned int n_explorers, QueenRoot *root_prefixes_h, 
	unsigned long long *vector_of_tree_size_h, unsigned long long *sols_h);
#ifdef __cplusplus
}
#endif


#endif
