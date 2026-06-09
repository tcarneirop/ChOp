#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <hip/hip_runtime.h>

#define _EMPTY_      -1
#include <vector>

// A tiny dummy kernel to force physical wave dispatch and queue wake-up
__global__ void dummy_warmup_kernel(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

void gpu_warmup_improved(int device_id) {
    float* d_warmup = nullptr;
    size_t allocation_size = 1024 * 1024 * sizeof(float); // 4MB to force actual MMU mapping

    // Force real physical memory allocation
    if (hipMalloc(&d_warmup, allocation_size) != hipSuccess) return;
    if (hipMemset(d_warmup, 0, allocation_size) != hipSuccess) return;

    // Wake up execution queues via actual kernel launch
    hipLaunchKernelGGL(dummy_warmup_kernel, dim3(1), dim3(64), 0, 0, d_warmup);
    
    hipDeviceSynchronize();
   
    hipFree(d_warmup);
}

void warmup_all_gpus(const int repetitions) {
    int num_devices = 0;
    if (hipGetDeviceCount(&num_devices) != hipSuccess || num_devices == 0) {
        return;
    }

    printf("\nFound %d HIP device(s). Initializing full context and P2P matrix...\n", num_devices);

    // Phase 1: Establish P2P connections explicitly to prevent runtime lag during the main execution
    for (int i = 0; i < num_devices; ++i) {
        hipSetDevice(i);
        for (int j = 0; j < num_devices; ++j) {
            if (i != j) {
                int can_access = 0;
                hipDeviceCanAccessPeer(&can_access, i, j);
                if (can_access) {
                    // This forces the driver to map memory spaces across Infinity Fabric/PCIe instantly
                    hipDeviceEnablePeerAccess(j, 0);
                }
            }
        }
    }

    // Phase 2: Warm up hardware queues across all devices sequentially
    for (int dev = 0; dev < num_devices; ++dev) {
        printf("Warming up GPU %d\n", dev);
        hipSetDevice(dev);

        for (int i = 0; i < repetitions; ++i) {
            gpu_warmup_improved(dev);
        }
    }

    // Restore primary device context state
    if (num_devices > 0) {
        hipSetDevice(0);
    }
}

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

inline void prefixesHandleSol(QueenRoot *root_prefixes, unsigned int flag,
															const char *board, int initialDepth, int num_sol)
{
	root_prefixes[num_sol].control = flag;
	for(int i = 0; i<initialDepth;++i)
		root_prefixes[num_sol].board[i] = board[i];
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


__device__  bool GPU_queens_stillLegal(const char *board, const int r){

	bool safe = true;
	int i, rev_i, offset;
	const char base = board[r];
	// Check vertical
	for ( i = 0, rev_i = r-1, offset=1; i < r; ++i, --rev_i, offset++)
		safe &= !((board[i] == base) | ( (board[rev_i] == base-offset) |
																		 (board[rev_i] == base+offset)));
	return safe;
}



__global__ void BP_queens_root_dfs(
	int N, unsigned int nPrefixes, int initial_depth,
	const QueenRoot *__restrict__ root_prefixes,
	unsigned long long *__restrict__ vector_of_tree_size,
	unsigned long long *__restrict__ sols)
{
	
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
               
        }else if ( MCstillLegal(board, depth) && !(flag &  bit_test ) ){//it is a valid subsol 

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


void nqueens(int size, int initial_depth, int block_size, unsigned int n_explorers, QueenRoot *root_prefixes_h ,
						 unsigned long long *vector_of_tree_size_h, unsigned long long *sols_h)
{


	unsigned long long *vector_of_tree_size_d;
	unsigned long long *sols_d;
	QueenRoot *root_prefixes_d;

	int num_blocks = ceil((double)n_explorers/block_size);


    hipMalloc((void**) &root_prefixes_d, n_explorers*sizeof(QueenRoot));
    hipMalloc((void**) &vector_of_tree_size_d, n_explorers*sizeof(unsigned long long));
    hipMalloc((void**) &sols_d, n_explorers*sizeof(unsigned long long));
    hipMemcpy(root_prefixes_d, root_prefixes_h, n_explorers * sizeof(QueenRoot), hipMemcpyHostToDevice);

	// warmup
	hipLaunchKernelGGL(BP_queens_root_dfs, num_blocks, block_size, 0, 0, 
		size,
		n_explorers,
		initial_depth,
		root_prefixes_d,
		vector_of_tree_size_d,
		sols_d);

	hipDeviceSynchronize();

	hipMemcpy(vector_of_tree_size_h, vector_of_tree_size_d, n_explorers*sizeof(unsigned long long), hipMemcpyDeviceToHost);
	hipMemcpy(sols_h, sols_d, n_explorers*sizeof(unsigned long long), hipMemcpyDeviceToHost);

//	hipFree(vector_of_tree_size_d);
//	hipFree(sols_d);
//	hipFree(root_prefixes_d);
	
}


int main(int argc, char *argv[])
{

    printf("\nStarting the warm up procedure\n");
  warmup_all_gpus(100);

  #ifdef IMPROVED
  printf("### IMPROVED SEARCH - Avoiding mirrored solutions\n");
  #endif



	if (argc != 4) {
		printf("Usage: %s <size> <initial depth> <block size>\n", argv[0]);
		return 1;
	}


	const int size = atoi(argv[1]);  // 17 - 15 for a short run
	const int initialDepth = atoi(argv[2]); // 6 or 7
	const int block_size = atoi(argv[3]); //32, 64, 128...
	

  unsigned long long initial_tree_size = 0ULL;
  unsigned long long qtd_sols_global = 0ULL;
  unsigned long long gpu_tree_size = 0ULL;

  unsigned int nMaxPrefixes = 75580635;

  printf("\n### Queens size: %d, Initial depth: %d, Block size: %d", size, initialDepth, block_size);


	QueenRoot* root_prefixes_h = (QueenRoot*)malloc(sizeof(QueenRoot)*nMaxPrefixes);
	unsigned long long *vector_of_tree_size_h = (unsigned long long*)malloc(sizeof(unsigned long long)*nMaxPrefixes);
	unsigned long long *solutions_h = (unsigned long long*)malloc(sizeof(unsigned long long)*nMaxPrefixes);

  double initial_time = rtclock();

	//initial search, getting Feasible, Valid and Incomplete solutions -- subproblems;
	unsigned long long n_explorers = BP_queens_prefixes(size, initialDepth, &initial_tree_size, root_prefixes_h);

	//calling the gpu-based search
	nqueens(size, initialDepth, block_size, n_explorers, root_prefixes_h, vector_of_tree_size_h, solutions_h);
  
  double final_time = rtclock();

  for(int i = 0; i<n_explorers;++i){
      qtd_sols_global += solutions_h[i];
      gpu_tree_size   += vector_of_tree_size_h[i];
  }

   #ifdef IMPROVED
       qtd_sols_global*=2;
    #endif


  printf("\nInitial tree size: %llu", initial_tree_size );
  printf("\nGPU Tree size: %llu\nTotal tree size: %llu\nNumber of solutions found: %llu.\n", gpu_tree_size,(initial_tree_size+gpu_tree_size),qtd_sols_global );
  printf("\nElapsed total: %.3f\n", (final_time-initial_time));
	

	free(root_prefixes_h);
	free(vector_of_tree_size_h);
	free(solutions_h);

	return 0;
}  
