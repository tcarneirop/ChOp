#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>

#include "../headers/timer.hpp"
#include "../headers/queens_subproblem.hpp"
#include "../headers/queens_CPU_GPU_subproblem_eval.hpp"
#include "../headers/queens_GPU_enumeration.hpp"
#include "../headers/queens_sub_gen.hpp"
#include "cuda_headers/cuda_funcs.hpp"
#include "../headers/queens_GPU_call_queens.hpp"





int main(int argc, char *argv[]){


    int block_size;
    int initialDepth;
    int size;

    #ifdef IMPROVED
        printf("### IMPROVED SEARCH - Avoiding mirrored solutions\n");
    #endif

    if (argc != 4) {
        printf("Usage: %s <size> <initial depth> <block size>\n", argv[0]);
        return 1;
    }

    size = atoi(argv[1]);
    initialDepth = atoi(argv[2]);
    block_size   =   atoi(argv[3]);

    SGPU_call_queens(size, initialDepth, block_size);

    return 0;
}
