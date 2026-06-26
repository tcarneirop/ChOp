
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

#include "../headers/queens.hpp"
#include "../headers/queens_GPU_enumeration.hpp"


#include "../headers/queens_omp_aux.hpp"
#include "../headers/helper.hpp"
#include "hip_headers/hip_warmup.hpp"
#include "hip_headers/hip_funcs.hpp"



void call_queens(int size, int initialDepth, int block_size){


    unsigned long long initial_tree_size = 0ULL;
    unsigned long long qtd_sols_global = 0ULL;
    unsigned long long gpu_tree_size = 0ULL;

    unsigned long long nMaxPrefixes = 75580635;
    int num_gpus = 0;

   
    hipGetDeviceCount( &num_gpus );
    printf("\nNumber of GPUS: %d\n", num_gpus );


    unsigned long long device_load[num_gpus];
  
    QueenRoot* root_prefixes_h = (QueenRoot*)malloc(sizeof(QueenRoot)*nMaxPrefixes);
    unsigned long long int *vector_of_tree_size_h = (unsigned long long int*)malloc(sizeof(unsigned long long int)*num_gpus);
    unsigned long long int *solutions_h = (unsigned long long int*)malloc(sizeof(unsigned long long int)*num_gpus);

    //calling the gpu-based search
    omp_set_num_threads(num_gpus);

    double initial_time = rtclock();
    

    //initial search, getting Feasible, Valid and Incomplete solutions -- subproblems;
    unsigned long long n_explorers =  queens_subproblem_generation((char)size, initialDepth ,&initial_tree_size, root_prefixes_h);

    printf("\n### Queens size: %d, Initial depth: %d, Block size: %d - Num_explorers: %llu", size, initialDepth, block_size,n_explorers);

    get_load_each_gpu(n_explorers, num_gpus, device_load);

    printf("\nLoad of each GPU:");
    for(int device = 0; device<num_gpus;++device){
        printf("\n\tDevice: %d - load : %llu ", device, device_load[device]);
    }
    printf("\n\n");

    #pragma omp parallel for default(none) shared(size, num_gpus, n_explorers, block_size, initialDepth, device_load, root_prefixes_h, vector_of_tree_size_h, solutions_h)
    for(uint device = 0; device<num_gpus; ++device){
  
        unsigned long long local_stride =  device * (n_explorers/num_gpus);
        //printf("\n\tNum threads: %d, Thread: %d - Device: %d - load : %llu ", omp_get_num_threads(),omp_get_thread_num(),  device, device_load[device]);
        HIP_call_queens(size, initialDepth,device_load[device], root_prefixes_h+local_stride,vector_of_tree_size_h+device, solutions_h+device, device, block_size);
    
    } 
 
    //Reducing the metrics
    for(int i = 0; i<num_gpus; ++i){
        qtd_sols_global += solutions_h[i];
        gpu_tree_size +=vector_of_tree_size_h[i];
    }
    
    double final_time = rtclock();

    #ifdef IMPROVED
    qtd_sols_global*=2;
    #endif


    printf("\nInitial tree size: %llu", initial_tree_size );
    printf("\nGPU Tree size: %llu\nTotal tree size: %llu\nNumber of solutions found: %llu\n", gpu_tree_size,(initial_tree_size+gpu_tree_size),qtd_sols_global );
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
