
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

#include "../headers/queens.hpp"
#include "../headers/queens_enumeration.hpp"


#include "../headers/queens_omp_aux.hpp"
#include "../headers/helper.hpp"
#include "hip_headers/hip_warmup.hpp"
#include "hip_headers/hip_funcs.hpp"



void call_queens(int size, int initialDepth, int block_size){


    unsigned long long initial_tree_size = 0ULL;
    unsigned long long qtd_sols_global = 0ULL;
    unsigned long long gpu_tree_size = 0ULL;

    unsigned int nMaxPrefixes = 75580635;
  

    double initial_time = rtclock();

  
    QueenRoot* root_prefixes_h = (QueenRoot*)malloc(sizeof(QueenRoot)*nMaxPrefixes);
    unsigned long long int *vector_of_tree_size_h = (unsigned long long int*)malloc(sizeof(unsigned long long int)*nMaxPrefixes);
    unsigned long long int *solutions_h = (unsigned long long int*)malloc(sizeof(unsigned long long int)*nMaxPrefixes);

    //initial search, getting Feasible, Valid and Incomplete solutions -- subproblems;
    unsigned long long n_explorers = queens_subproblem_generation((char)size, initialDepth ,&initial_tree_size, root_prefixes_h);

    printf("\n### Queens size: %d, Initial depth: %d, Block size: %d - Num_explorers: %llu", size, initialDepth, block_size,n_explorers);

    

    HIP_call_queens(size, initialDepth,n_explorers, root_prefixes_h,vector_of_tree_size_h, solutions_h, 0, block_size);
    
   
 
    //Reducing the metrics
    for(int i = 0; i<n_explorers;++i){
        qtd_sols_global += solutions_h[i];
        gpu_tree_size +=vector_of_tree_size_h[i];
    }
    
    double final_time = rtclock();

    #ifdef IMPROVED
    qtd_sols_global*=2;
    #endif

    printf("\nInitial tree size: %llu", initial_tree_size );
    printf("\nGPU Tree size: %llu\nTotal tree size: %llu\nNumber of solutions found: %llu.\n", gpu_tree_size,(initial_tree_size+gpu_tree_size),qtd_sols_global );
    printf("\nElapsed total: %.3f\n", (final_time-initial_time));

    
 
    #ifdef CHECKSOL
    if(qtd_sols_global == check_sols_number[size-1])
        printf("\n####### SUCCESS - CORRECT NUMBER OF SOLS. FOR SIZE %d\n", size);
    else
        printf("########## ERROR -- INCORRECT NUMBER FOS SOLS. FOR SIZE %d: %llu vs. %llu (correct)\n", size, qtd_sols_global,check_sols_number[size-1]);
    #endif


}


int main(int argc, char *argv[]){


    int block_size;
    int initialDepth;
    int size;

    #ifdef WARMUP
    warmup_all_gpus(100);
    #endif

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

    call_queens(size, initialDepth, block_size);

    return 0;
}
