
#ifndef QUEENS_SUB_HPP
#define QUEENS_SUB_HPP



inline void prefixesHandleSol(QueenRoot *root_prefixes, unsigned int flag, const char *board, const int initialDepth, const int num_sol)
{
	root_prefixes[num_sol].control = flag;
   
	for(int i = 0; i<initialDepth;++i)
		root_prefixes[num_sol].board[i] = board[i];
}

unsigned long long int queens_subproblem_generation(int size, int initialDepth,
    unsigned long long *tree_size, QueenRoot *root_prefixes){

    unsigned int flag = 0;
    int bit_test = 0;
    int8_t board[MAX_SIZE]; 
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
               
        }else if ( !(flag &  bit_test ) && queens_is_legal_placement(board, depth) ){//it is a valid subsol 

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
            	prefixesHandleSol(root_prefixes,flag,(char*)board,initialDepth,num_sol);
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