#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include <climits>



#include "../headers/helper.hpp"

#include "../headers/queens.hpp"
#include "../headers/queens_omp_aux.hpp"


void queens_omp_subtree_enumeration(const unsigned idx, const int N, const unsigned nPrefixes, 
    const int initial_depth, QueenRoot *__restrict__ root_prefixes,
    unsigned long long int *__restrict__ vector_of_tree_size, 
    unsigned long long int *__restrict__ sols){


    unsigned int flag = 0;
    int8_t board[MAX_SIZE];
    int N_l = N;
    int i, depth;
    unsigned long long  qtd_sols_thread = 0ULL;
    int depthGlobal = initial_depth;
    unsigned long long int tree_size = 0ULL;

    for (i = 0; i < N_l; ++i) {
        board[i] = EMPTY;
    }

    flag = root_prefixes[idx].control;

    for (i = 0; i < depthGlobal; ++i)
        board[i] = root_prefixes[idx].board[i];

    depth=initial_depth;

    do{

        //board[depth]++;
        const int mask = 1 << ++board[depth];

        if(board[depth] == N_l){

            board[depth] = EMPTY;
            //--depth;
            flag &= ~(1<<board[--depth]);
        }
        else{

            if (!(flag &  mask ) && queens_is_legal_placement(board, depth)){

                ++tree_size;
                flag |= mask;

                ++depth;

                if (depth == N_l) { //sol
                    ++qtd_sols_thread ;
                    --depth;
                    flag &= ~mask;
                }//if 
            }//else if
        }


    }while(depth >= depthGlobal); 

    sols[idx] = qtd_sols_thread ;
    vector_of_tree_size[idx] = tree_size;

}//kernel
////////


void call_queens(const int size, const int initialDepth){


    unsigned long long initial_tree_size = 0ULL;
    unsigned long long qtd_sols_global = 0ULL;
    unsigned long long total_tree_size = 0ULL;

    unsigned nMaxPrefixes = 95580635;
    int num_gpus = 0;


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
        queens_omp_subtree_enumeration(subproblem, size, n_subproblems, initialDepth, subproblems_pool,vector_of_tree_size_h, solutions_h);
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


int main(int argc, char *argv[]){


    int size;
    int initialDepth;

    #ifdef IMPROVED
        printf("### IMPROVED SEARCH - Avoiding mirrored solutions\n");
    #endif

    if (argc != 3) {
        printf("Usage: %s <size> <initial depth>\n", argv[0]);
        return 1;
    }

    size = atoi(argv[1]);
    initialDepth = atoi(argv[2]);

    call_queens(size, initialDepth);

    return 0;
}
