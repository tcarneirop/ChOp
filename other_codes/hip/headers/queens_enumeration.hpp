#ifndef QUEENS_ENUMERATION_HPP
#define QUEENS_ENUMERATION_HPP



__global__ void BP_queens_root_dfs(
	const int N, const unsigned int nPrefixes, const int initial_depth,
	QueenRoot *__restrict__ root_prefixes,
	unsigned long long *__restrict__ vector_of_tree_size,
	unsigned long long *__restrict__ sols)
{
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < nPrefixes) {
      unsigned int flag = 0;
      char board[24];
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

      depth=depthGlobal;

      do{

          board[depth]++;
          const int mask = 1<<board[depth];

          if(board[depth] == N_l){
              board[depth] = EMPTY;
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



#endif