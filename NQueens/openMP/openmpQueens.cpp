#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include <climits>



#include "../headers/timer.hpp"

#include "../headers/queens.hpp"
#include "../headers/queens_omp_aux.hpp"
#include "../headers/queens_general_enumeration.hpp"



#ifdef PERTHREAD

void REPORTcall_queens(const int size, const int initialDepth){


    unsigned long long initial_tree_size = 0ULL;
    unsigned long long qtd_sols_global = 0ULL;
    unsigned long long total_tree_size = 0ULL;

    unsigned nMaxPrefixes = 95580635;

    unsigned long long thread_load[omp_get_max_threads()];
    for(int i = 0; i<omp_get_max_threads();++i)
        thread_load[i] = 0ULL;
  
    QueenRoot* subproblems_pool = (QueenRoot*)malloc(sizeof(QueenRoot)*nMaxPrefixes);
    unsigned long long int *vector_of_tree_size_h = (unsigned long long int*)malloc(sizeof(unsigned long long int)*nMaxPrefixes);
    unsigned long long int *solutions_h = (unsigned long long int*)malloc(sizeof(unsigned long long int)*nMaxPrefixes);

    
    //initial search, getting Feasible, Valid and Incomplete solutions -- subproblems;
    unsigned long long n_subproblems = queens_subproblem_generation(size, initialDepth, &initial_tree_size, subproblems_pool);

   for(unsigned long long idx = 0; idx < n_subproblems;++idx) {
        volatile unsigned int t = subproblems_pool[idx].control;
        solutions_h[idx] = 0;
        vector_of_tree_size_h[idx] = 0;
    }

    double initial_time = rtclock();

    printf("\n### Queens size: %d, Initial depth: %d - Num_explorers: %llu - num_threads: %d", size, initialDepth,n_subproblems, omp_get_max_threads());

    #pragma omp parallel for schedule(runtime) default(none) shared(size, thread_load,n_subproblems, initialDepth, subproblems_pool, vector_of_tree_size_h, solutions_h)
    for(unsigned long long subproblem = 0; subproblem<n_subproblems; ++subproblem){
        int id = omp_get_thread_num();
        queens_default_subtree_enumeration(subproblem, size, n_subproblems, initialDepth, subproblems_pool,vector_of_tree_size_h, solutions_h);
        #ifdef REPORT
        thread_load[id] += vector_of_tree_size_h[subproblem];
        #endif
    } 
 
    //Reducing the metrics
    #pragma unroll 128
    for(unsigned long long i = 0; i<n_subproblems;++i)
        qtd_sols_global += solutions_h[i];
       
    #pragma unroll 128
    for(unsigned long long i = 0; i<n_subproblems;++i)
        total_tree_size +=vector_of_tree_size_h[i];

    double final_time = rtclock();

    #ifdef REPORT
    printf("\nThread load report: \n");
    unsigned long long biggest = 0;
    unsigned long long smallest = ULLONG_MAX;
    for(int id = 0; id<omp_get_max_threads();++id){
        if(thread_load[id]<smallest)
            smallest = thread_load[id];
        if(thread_load[id]>biggest)
            biggest = thread_load[id];
        printf("\t Thread Id: %d - Load: %llu - Percent: %.2f\n", id,thread_load[id],((double)thread_load[id]/(double)total_tree_size)*100.0);
    }
    #endif

    #ifdef IMPROVED
       qtd_sols_global*=2;
    #endif

    printf("\n\nInitial tree size: %llu", initial_tree_size );
    printf("\nParallel Tree size: %llu\nTotal tree size: %llu\nNumber of solutions found: %llu\n", total_tree_size,(initial_tree_size+total_tree_size),qtd_sols_global );
    printf("\nElapsed total: %.3f\n", (final_time-initial_time));
    #ifdef REPORT
    printf("\n\tBiggest thread load: %llu", biggest);
    printf("\n\tSmallest thread load: %llu", smallest);
    printf("\n\tBiggest/smallest: %.3fx\n", (double)biggest/(double)smallest);
    #endif


    #ifdef CHECKSOLS
    if(qtd_sols_global == check_sols_number[size-1])
        printf("\n####### SUCCESS - CORRECT NUMBER OF SOLS. FOR SIZE %d\n", size);
    else
        printf("########## ERROR -- INCORRECT NUMBER FOS SOLS. FOR SIZE %d: %llu vs. %llu (correct)\n", size, qtd_sols_global,check_sols_number[size-1]);
    #endif


}
#else
void call_queens(const int size, const int initialDepth){


    unsigned long long initial_tree_size = 0ULL;
    unsigned long long qtd_sols_global = 0ULL;
    unsigned long long total_tree_size = 0ULL;

    unsigned nMaxPrefixes = 95580635;

    QueenRoot* subproblems_pool = (QueenRoot*)malloc(sizeof(QueenRoot)*nMaxPrefixes);
  
    
    //initial search, getting Feasible, Valid and Incomplete solutions -- subproblems;
    unsigned long long n_subproblems = queens_subproblem_generation(size, initialDepth, &initial_tree_size, subproblems_pool);


    double initial_time = rtclock();

    printf("\n### Queens size: %d, Initial depth: %d - Num_explorers: %llu - num_threads: %d", size, initialDepth,n_subproblems, omp_get_max_threads());

    #pragma omp parallel for schedule(runtime) default(none) shared(size, n_subproblems, initialDepth, subproblems_pool) reduction(+:total_tree_size,qtd_sols_global)
    for(unsigned long long subproblem = 0; subproblem<n_subproblems; ++subproblem){
      
        unsigned long long l_treesize = 0ULL;
        unsigned long long l_sols =     0ULL;
        queens_default_subtree_enumeration(subproblem, size, n_subproblems, initialDepth, subproblems_pool,&l_treesize, &l_sols);
        total_tree_size+=l_treesize;
        qtd_sols_global+=l_sols;
    } 
 

    double final_time = rtclock();

    #ifdef IMPROVED
       qtd_sols_global*=2;
    #endif


    printf("\n\nInitial tree size: %llu", initial_tree_size );
    printf("\nParallel Tree size: %llu\nTotal tree size: %llu\nNumber of solutions found: %llu\n", total_tree_size,(initial_tree_size+total_tree_size),qtd_sols_global );
    printf("\nElapsed total: %.3f\n", (final_time-initial_time));
}
#endif



int main(int argc, char *argv[]){


    int size;
    int initialDepth;
    int block_size;
    #ifdef IMPROVED
        printf("### IMPROVED SEARCH - Avoiding mirrored solutions\n");
    #endif

    if (argc != 4) {
        printf("Usage: %s <size> <initial depth><block size>\n", argv[0]);
        return 1;
    }

    size = atoi(argv[1]);
    initialDepth = atoi(argv[2]);
    block_size   = atoi(argv[3]);
    
    #ifdef PERTHREAD
    printf("\n######## Version that does not use reduction and also provides a Per-thread load report ##########\n ");
    REPORTcall_queens(size, initialDepth);
    #else
    printf("\n######## Version that uses OMP reduction ##########\n ");
    call_queens(size,initialDepth);
    #endif
    return 0;
}
