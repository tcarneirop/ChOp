
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>


#define _EMPTY_      -1


double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

typedef struct queen_root{
    unsigned int control;
    int8_t board[12]; //maximum depth of the solution space.
} QueenRoot;


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(hipError_t code, const char *file, int line, bool abort=true)
{
    if (code != hipSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void  get_load_each_gpu(unsigned long long gpu_load, int num_gpus, unsigned long long *device_load){

    for(int device = 0; device<num_gpus;++device){
        device_load[device] = gpu_load/num_gpus;
        if(device == (num_gpus-1)){
            device_load[device]+= gpu_load%num_gpus;
        }
    }
}//////


__device__  inline bool GPU_queens_stillLegal(const char *__restrict__ board, const int r){

  bool safe = true;
  int i, rev_i, offset;
  const char base = board[r];
  // Check vertical
  for ( i = 0, rev_i = r-1, offset=1; i < r; ++i, --rev_i, offset++)
    safe &= !((board[i] == base) | ( (board[rev_i] == base-offset) |
                                     (board[rev_i] == base+offset)));
  return safe;
}

inline bool MCstillLegal(const char *board, const int r)
{

    int i;
    int ld;
    int rd;
    // Check vertical
    for ( i = 0; i < r; ++i)
        if (board[i] == board[r]) return false;
    // Check diagonals
    ld = board[r];  //left diagonal columns
    rd = board[r];  // right diagonal columns
    for ( i = r-1; i >= 0; --i) {
      --ld; ++rd;
      if (board[i] == ld || board[i] == rd) return false;
    }

    return true;
}


inline void prefixesHandleSol(QueenRoot *root_prefixes,unsigned int flag,char *board,int initialDepth,int num_sol){

    root_prefixes[num_sol].control = flag;

    for(int i = 0; i<initialDepth;++i)
      root_prefixes[num_sol].board[i] = (char)board[i];
}


unsigned long long int BP_queens_prefixes(int size, int initialDepth,
    unsigned long long *tree_size, QueenRoot *root_prefixes){

    unsigned int flag = 0;
    int bit_test = 0;
    char board[32]; 
    int i, depth; 
    unsigned long long int local_tree = 0ULL;
    unsigned long long int num_sol = 0;

    #ifdef IMPROVED
    uint break_cond =  (size/2) + (size & 1);
    #endif 

    /*initialization*/
    for (i = 0; i < size; ++i) { //
        board[i] = -1;
    }

    depth = 0;

    do{

        board[depth]++;
        bit_test = 0;
        bit_test |= (1<<board[depth]);


        if(board[depth] == size){
            board[depth] = _EMPTY_;
                //if(block_ub > upper)   block_ub = upper;
        }else if ( MCstillLegal(board, depth) && !(flag &  bit_test ) ){ //it is a valid subsol 
   
           #ifdef IMPROVED
            if(depth == 1){

                if(size& 1){
                    if (board[0] == break_cond-1 && board[1] > board[0]) 
                        break;
                }
                else{
                    if (board[0] == break_cond)
                        break;
                }
            }
            #endif 

                flag |= (1ULL<<board[depth]);
                depth++;
                ++local_tree;
                if (depth == initialDepth){ //handle solution
                   prefixesHandleSol(root_prefixes,flag,board,initialDepth,num_sol);
                   num_sol++;
            }else continue;
        }else continue;

        depth--;
        flag &= ~(1ULL<<board[depth]);

    }while(depth >= 0);

    *tree_size = local_tree;

    return num_sol;
}

__global__ void BP_queens_root_dfs( const int N, const unsigned int nPrefixes, 
    const int initial_depth,
    QueenRoot *__restrict__ root_prefixes,
    unsigned long long int *__restrict__ vector_of_tree_size, 
    unsigned long long int *__restrict__ sols){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
       
    if (idx < nPrefixes) {
        unsigned int flag = 0;
        char board[32];
        int N_l = N;
        int i, depth;
        unsigned long long  qtd_sols_thread = 0ULL;
        int depthGlobal = initial_depth;
        unsigned long long int tree_size = 0ULL;

        for (i = 0; i < N_l; ++i) {
            board[i] = _EMPTY_;
        }

        flag = root_prefixes[idx].control;

        for (i = 0; i < depthGlobal; ++i)
            board[i] = root_prefixes[idx].board[i];

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

                    if (depth == N_l) { //sol
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



void HIP_call_queens(int size, int initial_depth, unsigned int n_explorers, QueenRoot *root_prefixes_h ,
	unsigned long long *vector_of_tree_size_h, unsigned long long *sols_h, int gpu_id, int block_size){
     
   hipSetDevice(gpu_id);

    unsigned long long *vector_of_tree_size_d;
    unsigned long long *sols_d;
    QueenRoot *root_prefixes_d;

    int num_blocks = ceil((double)n_explorers/block_size);

    hipMalloc((void**) &vector_of_tree_size_d,n_explorers*sizeof(unsigned long long));
    hipMalloc((void**) &sols_d,n_explorers*sizeof(unsigned long long));
    hipMalloc((void**) &root_prefixes_d,n_explorers*sizeof(QueenRoot));

    hipMemcpy(root_prefixes_d, root_prefixes_h, n_explorers * sizeof(QueenRoot), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(BP_queens_root_dfs, num_blocks, block_size, 0, 0, 
        size,n_explorers,initial_depth,root_prefixes_d, vector_of_tree_size_d,sols_d);
    //BP_queens_root_dfs<<< num_blocks,block_size>>>(size,n_explorers,initial_depth,root_prefixes_d, vector_of_tree_size_d,sols_d);
   
    //gpuErrchk( cudaPeekAtLastError() );
    //hipDeviceSynchronize() ;

    hipMemcpy(vector_of_tree_size_h,vector_of_tree_size_d,n_explorers*sizeof(unsigned long long),hipMemcpyDeviceToHost);
    hipMemcpy(sols_h,sols_d,n_explorers*sizeof(unsigned long long),hipMemcpyDeviceToHost);
    
}

void call_queens(int size, int initialDepth, int block_size){


    unsigned long long initial_tree_size = 0ULL;
    unsigned long long qtd_sols_global = 0ULL;
    unsigned long long gpu_tree_size = 0ULL;

    unsigned int nMaxPrefixes = 75580635;
    int num_gpus = 0;

    hipGetDeviceCount( &num_gpus );
    printf("\nNumber of GPUS: %d\n", num_gpus );

    for(uint device = 0; device<num_gpus; ++device){
        hipSetDevice(device);
        hipFree(0);
    }
  


    unsigned long long device_load[num_gpus];
  
   
    QueenRoot* root_prefixes_h = (QueenRoot*)malloc(sizeof(QueenRoot)*nMaxPrefixes);
    unsigned long long int *vector_of_tree_size_h = (unsigned long long int*)malloc(sizeof(unsigned long long int)*nMaxPrefixes);
    unsigned long long int *solutions_h = (unsigned long long int*)malloc(sizeof(unsigned long long int)*nMaxPrefixes);

    //calling the gpu-based search
    omp_set_num_threads(num_gpus);

    double initial_time = rtclock();

    //initial search, getting Feasible, Valid and Incomplete solutions -- subproblems;
    unsigned long long n_explorers = BP_queens_prefixes((short)size, initialDepth ,&initial_tree_size, root_prefixes_h);

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
        HIP_call_queens(size, initialDepth,device_load[device], root_prefixes_h+local_stride,vector_of_tree_size_h+local_stride, solutions_h+local_stride, device, block_size);
    
    } 
 
    double final_time = rtclock();

    //Reducing the metrics
    for(int i = 0; i<n_explorers;++i){
        qtd_sols_global += solutions_h[i];
        gpu_tree_size +=vector_of_tree_size_h[i];
    }

    #ifdef IMPROVED
       qtd_sols_global*=2;
    #endif


    printf("\nInitial tree size: %llu", initial_tree_size );
    printf("\nGPU Tree size: %llu\nTotal tree size: %llu\nNumber of solutions found: %llu.\n", gpu_tree_size,(initial_tree_size+gpu_tree_size),qtd_sols_global );
    printf("\nElapsed total: %.3f\n", (final_time-initial_time));

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
