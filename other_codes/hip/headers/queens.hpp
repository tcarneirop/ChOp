
#ifndef QUEENS_HPP
#define QUEENS_HPP


#define EMPTY        -1
#define MAX_SIZE        24
#define SUBPROBLEM_SIZE 10


typedef struct queen_root{
		unsigned int control;
		int8_t board[SUBPROBLEM_SIZE]; //maximum depth of the solution space.
} QueenRoot;

inline void prefixesHandleSol(QueenRoot *__restrict__ root_prefixes,unsigned int flag,char *__restrict__ board,int initialDepth,int num_sol){

    root_prefixes[num_sol].control = flag;

    for(int i = 0; i<initialDepth;++i)
      root_prefixes[num_sol].board[i] = (char)board[i];
}


__device__ __host__ inline bool GPU_queens_stillLegal(const char *__restrict__  board, const int r){

	bool safe = true;
	int i, rev_i, offset;
	const char base = board[r];
	// Check vertical
	for ( i = 0, rev_i = r-1, offset=1; i < r; ++i, --rev_i, offset++)
		safe &= !((board[i] == base) | ( (board[rev_i] == base-offset) |(board[rev_i] == base+offset)));
	return safe;
}

unsigned long long int BP_queens_prefixes(int size, int initialDepth,
    unsigned long long *tree_size, QueenRoot *root_prefixes){

    unsigned int flag = 0;
    int bit_test = 0;
    char board[MAX_SIZE]; 
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
            board[depth] = EMPTY;
               
        }else if ( !(flag &  bit_test ) && GPU_queens_stillLegal(board, depth) ){//it is a valid subsol 

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



#endif