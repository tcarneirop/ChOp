#ifndef QUEENS_GPU_CALL_QUEENS
#define QUEENS_GPU_CALL_QUEENS


void SGPU_call_queens(int size, int initialDepth, int block_size){


    unsigned long long initial_tree_size = 0ULL;
    unsigned long long qtd_sols_global = 0ULL;
    unsigned long long gpu_tree_size = 0ULL;

    unsigned long long nMaxPrefixes = 75580635;

    QueenRoot* root_prefixes_h = (QueenRoot*)malloc(sizeof(QueenRoot)*nMaxPrefixes);
    unsigned long long vector_of_tree_size_h = 0ULL;
    unsigned long long solutions_h =  0ULL;

    //initial search, getting Feasible, Valid and Incomplete solutions -- subproblems;
    unsigned long long n_explorers = queens_subproblem_generation((char)size, initialDepth ,&initial_tree_size, root_prefixes_h);

    printf("\n### Queens size: %d, Initial depth: %d, Block size: %d - Num_explorers: %llu", size, initialDepth, block_size,n_explorers);

    double initial_time = rtclock();

    #if defined(__HIPCC__)
    HIP_call_queens(size, initialDepth,n_explorers, root_prefixes_h,&vector_of_tree_size_h, &solutions_h, 0, block_size);   
    #elif defined(__CUDACC__)
    CUDA_call_queens(size, initialDepth,n_explorers, root_prefixes_h,&vector_of_tree_size_h, &solutions_h, 0, block_size);   
    #else
    printf("########## COMPILATION ERROR: HIPCC/CUDACC not defined ############");
    exit(EXIT_FAILURE);
    #endif

    //Reducing the metrics
    gpu_tree_size   += vector_of_tree_size_h;
    qtd_sols_global += solutions_h;

    double final_time = rtclock();

    #ifdef IMPROVED
    qtd_sols_global*=2;
    #endif

    printf("\nInitial tree size: %llu", initial_tree_size );
    printf("\nGPU Tree size: %llu\nTotal tree size: %llu\nNumber of solutions found: %llu\n", gpu_tree_size,(initial_tree_size+gpu_tree_size),qtd_sols_global );
    printf("\nElapsed total: %.3f\n", (final_time-initial_time));

    #ifdef CHECKSOLS
    if(qtd_sols_global == check_sols_number[size-1])
        printf("\n####### SUCCESS - CORRECT NUMBER OF SOLS. FOR SIZE %d\n", size);
    else
        printf("########## ERROR -- INCORRECT NUMBER FOS SOLS. FOR SIZE %d: %llu vs. %llu (correct)\n", size, qtd_sols_global,check_sols_number[size-1]);
    #endif

}


#endif
