#define MAX 64

#define _EMPTY_      -1

#include <stdio.h>
#include <stdlib.h>


bool stillLegal(const int *board, const int r)
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


unsigned long long  BP_queens_serial(const int size, unsigned long long *tree_size){

    unsigned int flag = 0U;
    unsigned int bit_test = 0U;
    int board[MAX]; //representa o ciclo
    int i, depth; //para dizer que 0-1 ja foi visitado e a busca comeca de 1, bote 2
    unsigned long long int local_tree = 0ULL;
    unsigned int num_sol = 0;
   
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
        }else if ( stillLegal(board, depth) && !(flag &  bit_test ) ){ //is legal

                flag |= (1ULL<<board[depth]);
                depth++;
                ++local_tree;
                if (depth == size){ //handle solution
                   // handleSolution(board,size);
                   num_sol++;
            }else continue;
        }else continue;

        depth--;
        flag &= ~(1ULL<<board[depth]);

    }while(depth >= 0);

    *tree_size = local_tree;

    return num_sol;
}


int main(int argc, char *argv[]){


    int size = atoi(argv[1]);

    unsigned long long tree_size = 0ULL;
    unsigned long long nsol = BP_queens_serial(size,&tree_size);

    printf("Queens of Size: %d \n\t Number of solutions found: %llu \n\t Tree size: %llu\n",size, nsol,tree_size);

    return 0;
}      


