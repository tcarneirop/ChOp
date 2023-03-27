#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <hip/hip_runtime.h>

#define _EMPTY_      -1


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
	int N, unsigned int nPreFixos, int depthPreFixos,
	const QueenRoot *__restrict__ root_prefixes,
	unsigned long long *__restrict__ vector_of_tree_size,
	unsigned long long *__restrict__ sols)
{
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < nPrefixes) {
      unsigned int flag = 0;
      unsigned int bit_test = 0;
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
						 unsigned long long *vector_of_tree_size_h, unsigned long long *sols_h, const int repeat)
{


	unsigned long long *vector_of_tree_size_d;
	unsigned long long *sols_d;
	QueenRoot *root_prefixes_d;

	int num_blocks = ceil((double)n_explorers/block_size);

	hipMalloc((void**) &vector_of_tree_size_d, n_explorers*sizeof(unsigned long long));
	hipMalloc((void**) &sols_d, n_explorers*sizeof(unsigned long long));
	hipMalloc((void**) &root_prefixes_d, n_explorers*sizeof(QueenRoot));
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
	auto start = std::chrono::steady_clock::now();

	for (int i = 0; i < repeat; i++) {
		hipLaunchKernelGGL(BP_queens_root_dfs, num_blocks, _QUEENS_BLOCK_SIZE_, 0, 0, 
			size,
			n_explorers,
			initial_depth,
			root_prefixes_d,
			vector_of_tree_size_d,
			sols_d);
	}

	hipDeviceSynchronize();
	auto end = std::chrono::steady_clock::now();
	auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
	printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

	hipMemcpy(vector_of_tree_size_h, vector_of_tree_size_d, n_explorers*sizeof(unsigned long long), hipMemcpyDeviceToHost);
	hipMemcpy(sols_h, sols_d, n_explorers*sizeof(unsigned long long), hipMemcpyDeviceToHost);

	hipFree(vector_of_tree_size_d);
	hipFree(sols_d);
	hipFree(root_prefixes_d);
	
}


int main(int argc, char *argv[])
{


	hipFree(0);


	if (argc != 4) {
		printf("Usage: %s <size> <initial depth> <block size> <repeat>\n", argv[0]);
		return 1;
	}


	const int size = atoi(argv[1]);  // 17 - 15 for a short run
	const int initialDepth = atoi(argv[2]); // 6 or 7
	const int block_size = atoi(argv[3]); //32, 64, 128...
	const int repeat = atoi(argv[4]); // kernel execution times
	


  unsigned long long initial_tree_size = 0ULL;
  unsigned long long qtd_sols_global = 0ULL;
  unsigned long long gpu_tree_size = 0ULL;

  unsigned int nMaxPrefixes = 75580635;

  printf("\n### Queens size: %d, Initial depth: %d, Block size: %d, Repeat: %d", size, initialDepth, block_size, repeat);


	QueenRoot* root_prefixes_h = (QueenRoot*)malloc(sizeof(QueenRoot)*nMaxPrefixos);
	unsigned long long *vector_of_tree_size_h = (unsigned long long*)malloc(sizeof(unsigned long long)*nMaxPrefixos);
	unsigned long long *solutions_h = (unsigned long long*)malloc(sizeof(unsigned long long)*nMaxPrefixos);


	if (root_prefixes_h == NULL || vector_of_tree_size_h == NULL || solutions_h == NULL) {
		printf("Error: host out of memory\n");
		if (root_prefixes_h) free(root_prefixes_h);
		if (vector_of_tree_size_h) free(vector_of_tree_size_h);
		if (solutions_h) free(solutions_h);
		return 1;
	}

	//initial search, getting Feasible, Valid and Incomplete solutions -- subproblems;
	unsigned long long n_explorers = BP_queens_prefixes(size, initialDepth, &initial_tree_size, root_prefixes_h);

	//calling the gpu-based search
	nqueens(size, initialDepth, block_size, n_explorers, root_prefixes_h, vector_of_tree_size_h, solutions_h, repeat);

  for(int i = 0; i<n_explorers;++i){
      qtd_sols_global += solutions_h[i];
      gpu_tree_size   += vector_of_tree_size_h[i];
  }

  printf("\nInitial tree size: %llu", initial_tree_size );
  printf("\nGPU Tree size: %llu\nTotal tree size: %llu\nNumber of solutions found: %llu.\n", gpu_tree_size,(initial_tree_size+gpu_tree_size),qtd_sols_global );
  printf("\nElapsed total: %.3f\n", (final_time-initial_time));
	
	// Initial depth: 7 - Size: 15:
	// Tree size: 2466109
	// Number of solutions found: 2279184
	// Tree size: 171129071
	if (size == 15 && initialDepth == 7) {
		if (qtd_sols_global == 2279184 && tree_size == 171129071)
			printf("PASS\n");
		else
			printf("FAIL\n");
	}

	free(root_prefixes_h);
	free(vector_of_tree_size_h);
	free(solutions_h);

	return 0;
}  
