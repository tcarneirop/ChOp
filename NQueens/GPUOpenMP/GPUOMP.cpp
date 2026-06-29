#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include <climits>



#include "../headers/timer.hpp"
#include "../headers/queens_subproblem.hpp"
#include "../headers/queens_CPU_GPU_subproblem_eval.hpp"
#include "../headers/queens_default_enumeration.hpp"
#include "../headers/queens_sub_gen.hpp"


void OMP_SGPU_call_queens(const int size, const int initialDepth, const int block_size){


    unsigned long long initial_tree_size = 0ULL;
    unsigned long long qtd_sols_global = 0ULL;
    unsigned long long total_tree_size = 0ULL;

    unsigned nMaxPrefixes = 95580635;

    QueenRoot* subproblems_pool = (QueenRoot*)malloc(sizeof(QueenRoot)*nMaxPrefixes);
    //initial search, getting Feasible, Valid and Incomplete solutions -- subproblems;
    unsigned long long n_subproblems = queens_subproblem_generation(size, initialDepth, &initial_tree_size, subproblems_pool);
    
    unsigned long long * vector_of_tree_size_h = (unsigned long long *)malloc(sizeof(unsigned long long)*n_subproblems);
    unsigned long long * sols_h =  (unsigned long long *)malloc(sizeof(unsigned long long)*n_subproblems);

    double initial_time = rtclock();

    printf("\n### Queens size: %d, Initial depth: %d - Num_explorers: %llu - block_size: %d", size, initialDepth,n_subproblems, omp_get_max_threads());

    #pragma omp target data map (to: subproblems_pool[0:n_subproblems]) \
                        map (from: vector_of_tree_size_h[0:n_subproblems ], \
                                   sols_h[0:n_subproblems ])
    {
    
        #pragma omp target teams distribute parallel for thread_limit(block_size)
        for(unsigned long long subproblem = 0; subproblem<n_subproblems; ++subproblem){
            queens_default_subtree_enumeration(subproblem, size, n_subproblems, initialDepth, subproblems_pool,vector_of_tree_size_h, sols_h);
        } 
    
    }

    for(unsigned long long i = 0;i<n_subproblems;++i){
        qtd_sols_global+=sols_h[i];
        total_tree_size+=vector_of_tree_size_h[i];
    }
    

    double final_time = rtclock();

    #ifdef IMPROVED
       qtd_sols_global*=2;
    #endif


    printf("\n\nInitial tree size: %llu", initial_tree_size );
    printf("\nParallel Tree size: %llu\nTotal tree size: %llu\nNumber of solutions found: %llu\n", total_tree_size,(initial_tree_size+total_tree_size),qtd_sols_global );
    printf("\nElapsed total: %.3f\n", (final_time-initial_time));
}



int main(int argc, char *argv[]){


    int size;
    int initialDepth;
    int block_size;

    #ifdef IMPROVED
        printf("### IMPROVED SEARCH - Avoiding mirrored solutions\n");
    #endif

    if (argc != 4) {
        printf("Usage: %s <size> <initial depth><block_size>\n", argv[0]);
        return 1;
    }

    size = atoi(argv[1]);
    initialDepth = atoi(argv[2]);
    block_size = atoi(argv[3]);
    OMP_SGPU_call_queens(size,initialDepth,block_size);
    return 0;
}
