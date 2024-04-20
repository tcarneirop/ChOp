#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <hip/hip_runtime.h>

#include "../headers/AMD_queens.h"

double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
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

int main(int argc, char *argv[])
{


	hipFree(0);

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



	if (root_prefixes_h == NULL || vector_of_tree_size_h == NULL || solutions_h == NULL) {
		printf("Error: host out of memory\n");
		if (root_prefixes_h) free(root_prefixes_h);
		if (vector_of_tree_size_h) free(vector_of_tree_size_h);
		if (solutions_h) free(solutions_h);
		return 1;
	}



  double initial_time = rtclock();

	//initial search, getting Feasible, Valid and Incomplete solutions -- subproblems;
	unsigned long long n_explorers = BP_queens_prefixes(size, initialDepth, &initial_tree_size, root_prefixes_h);

  AMD_call_cuda_queens(size, initialDepth, n_explorers, root_prefixes_h, vector_of_tree_size_h, solutions_h, 0);

  double final_time = rtclock();

  for(int i = 0; i<n_explorers;++i){
      qtd_sols_global += solutions_h[i];
      gpu_tree_size   += vector_of_tree_size_h[i];
  }

  printf("\nInitial tree size: %llu", initial_tree_size );
  printf("\nGPU Tree size: %llu\nTotal tree size: %llu\nNumber of solutions found: %llu.\n", gpu_tree_size,(initial_tree_size+gpu_tree_size),qtd_sols_global );
  printf("\nElapsed total: %.3f\n", (final_time-initial_time));
	

	free(root_prefixes_h);
	free(vector_of_tree_size_h);
	free(solutions_h);

	return 0;
}  
