
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <cmath>
#include <omp.h>
#include <sys/time.h>

#ifndef MAX_BOARDSIZE
    #define  MAX_BOARDSIZE 24
#endif 

typedef unsigned long long SOLUTIONTYPE;

#define MIN_BOARDSIZE 2

SOLUTIONTYPE g_numsolutions = 0ULL;


double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

typedef struct subproblem{
    
    long long  aQueenBitRes[MAX_BOARDSIZE]; /* results */
    long long  aQueenBitCol[MAX_BOARDSIZE]; /* marks colummns which already have queens */
    long long  aQueenBitPosDiag[MAX_BOARDSIZE]; /* marks "positive diagonals" which already have queens */
    long long  aQueenBitNegDiag[MAX_BOARDSIZE]; /* marks "negative diagonals" which already have queens */
    long long  subproblem_stack[MAX_BOARDSIZE+2]; /* we use a stack instead of recursion */
    long long   pnStackPos;
    unsigned long long num_sols_sub;
} Subproblem;


unsigned long long partial_search_64(long long board_size, long long cutoff_depth, Subproblem *__restrict__ subproblem_pool)
{

    long long aQueenBitRes[MAX_BOARDSIZE]; /* results */
    long long aQueenBitCol[MAX_BOARDSIZE]; /* marks colummns which already have queens */
    long long aQueenBitPosDiag[MAX_BOARDSIZE]; /* marks "positive diagonals" which already have queens */
    long long aQueenBitNegDiag[MAX_BOARDSIZE]; /* marks "negative diagonals" which already have queens */
    long long aStack[MAX_BOARDSIZE + 2]; /* we use a stack instead of recursion */
     
    register long long int *pnStack;

    register long long int pnStackPos = 0LLU;

    register long long numrows = 0LL; /* numrows redundant - could use stack */
    register unsigned long long lsb; /* least significant bit */
    register unsigned long long bitfield; /* bits which are set mark possible positions for a queen */
    long long i;
    long long odd = board_size & 1LL; /* 0 if board_size even, 1 if odd */
    
    //Change here for the pool
    //long long int board_minus = 45LL; /* board size - 1 */
    long long mask = (1LL << board_size) - 1LL; /* if board size is N, mask consists of N 1's */

    unsigned long long tree_size = 0ULL;
    /* Initialize stack */
    aStack[0] = -1LL; /* set sentinel -- signifies end of stack */

    /* NOTE: (board_size & 1) is true iff board_size is odd */
    /* We need to loop through 2x if board_size is odd */
    for (i = 0; i < (1 + odd); ++i)
    {
        /* We don't have to optimize this part; it ain't the
           critical loop */
        bitfield = 0ULL;
        if (0LL == i)
        {
            /* Handle half of the board, except the middle
               column. So if the board is 5 x 5, the first
               row will be: 00011, since we're not worrying
               about placing a queen in the center column (yet).
            */
            long long int half = board_size>>1LL; /* divide by two */
            /* fill in rightmost 1's in bitfield for half of board_size
               If board_size is 7, half of that is 3 (we're discarding the remainder)
               and bitfield will be set to 111 in binary. */
            bitfield = (1LL << half) - 1LL;
            pnStack = aStack + 1LL; /* stack pointer */
            
            pnStackPos++;

            aQueenBitRes[0] = 0LL;
            aQueenBitCol[0] = aQueenBitPosDiag[0] = aQueenBitNegDiag[0] = 0LL;

        }
        else
        {
            /* Handle the middle column (of a odd-sized board).
               Set middle column bit to 1, then set
               half of next row.
               So we're processing first row (one element) & half of next.
               So if the board is 5 x 5, the first row will be: 00100, and
               the next row will be 00011.
            */
            bitfield = 1 << (board_size >> 1);
            numrows = 1; /* prob. already 0 */

            /* The first row just has one queen (in the middle column).*/
            aQueenBitRes[0] = bitfield;
            aQueenBitCol[0] = aQueenBitPosDiag[0] = aQueenBitNegDiag[0] = 0LL;
            aQueenBitCol[1] = bitfield;

            /* Now do the next row.  Only set bits in half of it, because we'll
               flip the results over the "Y-axis".  */
            aQueenBitNegDiag[1] = (bitfield >> 1ULL);
            aQueenBitPosDiag[1] = (bitfield << 1ULL);
            pnStack = aStack + 1LL; /* stack pointer */
            
            pnStackPos++;

            *pnStack++ = 0LL; /* we're done w/ this row -- only 1 element & we've done it */
            bitfield = (bitfield - 1ULL) >> 1ULL; /* bitfield -1 is all 1's to the left of the single 1 */
        }

        /* this is the critical loop */
        for (;;)
        {
         
            lsb = -((signed long long)bitfield) & bitfield; /* this assumes a 2's complement architecture */
            
            if (0ULL == bitfield)
            {
                
                bitfield = *--pnStack; /* get prev. bitfield from stack */
                pnStackPos--;

                if (pnStack == aStack) { /* if sentinel hit.... */
                    break ;
                }
                
                --numrows;
                continue;
            }
            
            bitfield &= ~lsb; /* toggle off this bit so we don't try it again */
            aQueenBitRes[numrows] = lsb; /* save the result */

            if (numrows < cutoff_depth) /* we still have more rows to process? */
            {
                long long n = numrows++;
                aQueenBitCol[numrows] = aQueenBitCol[n] | lsb;
                aQueenBitNegDiag[numrows] = (aQueenBitNegDiag[n] | lsb) >> 1LL;
                aQueenBitPosDiag[numrows] = (aQueenBitPosDiag[n] | lsb) << 1LL;

                pnStackPos++;
                
                *pnStack++ = bitfield;
                
                bitfield = mask & ~(aQueenBitCol[numrows] | aQueenBitNegDiag[numrows] | aQueenBitPosDiag[numrows]);
                
                ++tree_size;

                if(numrows == cutoff_depth){
                 
                    memcpy(subproblem_pool[g_numsolutions].aQueenBitRes, aQueenBitRes, sizeof(long long)*MAX_BOARDSIZE);
                    memcpy(subproblem_pool[g_numsolutions].aQueenBitCol, aQueenBitCol, sizeof(long long)*MAX_BOARDSIZE);
                    memcpy(subproblem_pool[g_numsolutions].aQueenBitPosDiag, aQueenBitPosDiag, sizeof(long long)*MAX_BOARDSIZE);
                    memcpy(subproblem_pool[g_numsolutions].aQueenBitNegDiag, aQueenBitNegDiag, sizeof(long long)*MAX_BOARDSIZE);
                    memcpy(subproblem_pool[g_numsolutions].subproblem_stack, aStack, sizeof(long long)*(MAX_BOARDSIZE+2));
                   
                    ++g_numsolutions;

                } //if partial solution

                continue;
            }
            else
            {
                  
                bitfield = *--pnStack;
                pnStackPos--;
                --numrows;   
                continue;
            }
        }
    }

    return tree_size;

}



void mcore_final_search(long long board_size, long long cutoff_depth, Subproblem * subproblem, int index,
    unsigned long long *__restrict__ vec_tree_size, unsigned long long *__restrict__ vec_num_sols)
{

    long long* aQueenBitRes = subproblem->aQueenBitRes; 
    long long* aQueenBitCol = subproblem->aQueenBitCol; 
    long long* aQueenBitPosDiag = subproblem->aQueenBitPosDiag; 
    long long* aQueenBitNegDiag = subproblem->aQueenBitNegDiag ; 
    long long* aStack = subproblem->subproblem_stack ; 

    register long long *pnStack;
    
    //long long int pnStackPos = subproblem->pnStackPos;

    long long int board_minus = board_size - 1LL; /* board size - 1 */
    long long int mask = (1LL << board_size) - 1LL; /* if board size is N, mask consists of N 1's */

    unsigned long long local_num_sols = 0ULL;
    unsigned long long tree_size = 0ULL;
    
    register unsigned long long lsb; 
    register unsigned long long bitfield; 

    
    register long long numrows = cutoff_depth;

    //pnStack = aStack + pnStackPos; /* stack pointer */
    pnStack = aStack; /* stack pointer */
    
    bitfield = mask & ~(aQueenBitCol[numrows] | aQueenBitNegDiag[numrows] | aQueenBitPosDiag[numrows]);
    
    /* this is the critical loop */
    for (;;)
    {
    
        lsb = -((signed long long)bitfield) & bitfield; /* this assumes a 2's complement architecture */
        
        if (0ULL == bitfield)
        {
           
            if(numrows <= cutoff_depth){ 
                //printf("\nEND OF THE SUBPROBLEM EXPLORATION! %d", numrows);
                break ;
            }

            bitfield = *--pnStack; /* get prev. bitfield from stack */
            
            //printf("Backtracking!");
            --numrows;
            continue;
        }

        bitfield &= ~lsb; /* toggle off this bit so we don't try it again */

        aQueenBitRes[numrows] = lsb; /* save the result */
        
        if (numrows < board_minus) /* we still have more rows to process? */
        {
        
            long long n = numrows++;
            aQueenBitCol[numrows] = aQueenBitCol[n] | lsb;
            aQueenBitNegDiag[numrows] = (aQueenBitNegDiag[n] | lsb) >> 1LL;
            aQueenBitPosDiag[numrows] = (aQueenBitPosDiag[n] | lsb) << 1LL;
            *pnStack++ = bitfield;

            bitfield = mask & ~(aQueenBitCol[numrows] | aQueenBitNegDiag[numrows] | aQueenBitPosDiag[numrows]);
            ++tree_size;
            continue;
        }
        else
        {

            ++local_num_sols;
            bitfield = *--pnStack;
            --numrows;
            continue;
        }
    }

    //returning the number of solutions
    vec_num_sols[index] = local_num_sols;
    vec_tree_size[index] = tree_size;
}




void call_mcore_search(long long board_size, long long cutoff_depth){
    
    
    double initial_time = rtclock();

    Subproblem *subproblem_pool = (Subproblem*)(malloc(sizeof(Subproblem)*(unsigned)10000000));
    
    g_numsolutions = 0ULL;
    
    unsigned long long initial_tree_size = partial_search_64(board_size,cutoff_depth, subproblem_pool);
    
    unsigned long long num_subproblems = g_numsolutions;
    
    unsigned long long num_sols_search = 0ULL;
    unsigned long long mcore_tree_size[num_subproblems];
    unsigned long long mcore_num_sols[num_subproblems];
    unsigned long long total_mcore_num_sols = 0ULL;
    unsigned long long total_mcore_tree_size = 0ULL;

    for(unsigned long long i = 0; i<num_subproblems;++i){
        mcore_num_sols[i] = 0ULL;
        mcore_tree_size[i] = 0ULL;
    }

    printf("\nPartial tree: %llu -- Number of subproblems: %llu \n", initial_tree_size, num_subproblems);
    printf("\n### MCORE Search ###\n\tNumber of subproblems: %lld - Size: %lld, Initial depth: %lld,  Num threads: %d\n", num_subproblems, board_size, cutoff_depth, omp_get_num_procs());
    
    #pragma omp parallel for schedule(runtime) default(none) shared(num_subproblems,board_size,mcore_tree_size,mcore_num_sols, cutoff_depth, subproblem_pool)
    for(int s = 0; s<num_subproblems; ++s){
        mcore_final_search(board_size, cutoff_depth, subproblem_pool+s, s,mcore_tree_size,mcore_num_sols);
    }

    for(int s = 0; s<num_subproblems;++s){
        total_mcore_tree_size+=mcore_tree_size[s];
        total_mcore_num_sols+=mcore_num_sols[s];
    }


    printf("\nPARALLEL SEARCH: size %lld, Tree: %llu,  solutions: %llu\n", board_size,total_mcore_tree_size+initial_tree_size, total_mcore_num_sols*2);
    
    printf("\n#######################################\n");

    double final_time = rtclock();
    printf("\nElapsed total: %.3f\n", (final_time-initial_time));

    free(subproblem_pool);

}////////////////////////////////////////////////




/* main routine for N Queens program.*/
int main(int argc, char** argv)
{

    int boardsize = atoi(argv[1]);
        //exec, size, search, depth, chunk; 
        call_mcore_search(boardsize, (long long)(atoi(argv[2])));
    return 0;
}



