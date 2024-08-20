
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
    
    long long  aQueenBitRes; 
    long long  aQueenBitCol;
    long long  aQueenBitPosDiag;
    long long  aQueenBitNegDiag; 
} Subproblem;



unsigned long long partial_search_64(long long board_size, long long cutoff_depth, Subproblem *__restrict__ subproblem_pool)
{

    long long aQueenBitRes[MAX_BOARDSIZE];
    long long aQueenBitCol[MAX_BOARDSIZE]; 
    long long aQueenBitPosDiag[MAX_BOARDSIZE]; 
    long long aQueenBitNegDiag[MAX_BOARDSIZE]; 
    long long aStack[MAX_BOARDSIZE]; 
     
    register long long int *pnStack;

    register long long int pnStackPos = 0LLU;

    register long long numrows = 0LL; 
    register unsigned long long lsb; 
    register unsigned long long bitfield; 
    long long i;
    long long odd = board_size & 1LL; 
    
    long long mask = (1LL << board_size) - 1LL; 
    
    unsigned long long tree_size = 0ULL;
    /* Initialize stack */
    aStack[0] = -1LL; /* set sentinel -- signifies end of stack */

    /* NOTE: (board_size & 1) is true iff board_size is odd */
    /* We need to loop through 2x if board_size is odd */
    for (i = 0; i < (1 + odd); ++i)
    {
        bitfield = 0ULL;
        if (0LL == i)
        {
           
            long long int half = board_size>>1LL; 
            bitfield = (1LL << half) - 1LL;
            pnStack = aStack + 1LL; 
            
            pnStackPos++;

            aQueenBitRes[0] = 0LL;
            aQueenBitCol[0] = aQueenBitPosDiag[0] = aQueenBitNegDiag[0] = 0LL;

        }
        else
        {
            bitfield = 1 << (board_size >> 1);
            numrows = 1; /* prob. already 0 */

            aQueenBitRes[0] = bitfield;
            aQueenBitCol[0] = aQueenBitPosDiag[0] = aQueenBitNegDiag[0] = 0LL;
            aQueenBitCol[1] = bitfield;

            aQueenBitNegDiag[1] = (bitfield >> 1ULL);
            aQueenBitPosDiag[1] = (bitfield << 1ULL);
            pnStack = aStack + 1LL;
            
            pnStackPos++;

            *pnStack++ = 0LL;
            bitfield = (bitfield - 1ULL) >> 1ULL; 
        }

        for (;;)
        {
         
            lsb = -((signed long long)bitfield) & bitfield; 
            if (0ULL == bitfield)
            {
                
                bitfield = *--pnStack; 
                pnStackPos--;

                if (pnStack == aStack) { 
                    break ;
                }
                
                --numrows;
                continue;
            }
            
            bitfield &= ~lsb; 
            aQueenBitRes[numrows] = lsb; 
            
            if (numrows < cutoff_depth) {
                long long n = numrows++;
                aQueenBitCol[numrows] = aQueenBitCol[n] | lsb;
                aQueenBitNegDiag[numrows] = (aQueenBitNegDiag[n] | lsb) >> 1LL;
                aQueenBitPosDiag[numrows] = (aQueenBitPosDiag[n] | lsb) << 1LL;

                pnStackPos++;
                
                *pnStack++ = bitfield;
                
                bitfield = mask & ~(aQueenBitCol[numrows] | aQueenBitNegDiag[numrows] | aQueenBitPosDiag[numrows]);
                
                ++tree_size;

                if(numrows == cutoff_depth){
                 
                    subproblem_pool[g_numsolutions].aQueenBitRes =  aQueenBitRes[numrows];
                    subproblem_pool[g_numsolutions].aQueenBitCol = aQueenBitCol[numrows];
                    subproblem_pool[g_numsolutions].aQueenBitPosDiag = aQueenBitPosDiag[numrows];
                    subproblem_pool[g_numsolutions].aQueenBitNegDiag = aQueenBitNegDiag[numrows];
                    
                    ++g_numsolutions;

                } 
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



void mcore_final_search(long long board_size, long long cutoff_depth, Subproblem *__restrict__ subproblem, int index,
    unsigned long long *__restrict__ vec_tree_size, unsigned long long *__restrict__ vec_num_sols)
{

    
    long long aQueenBitRes[MAX_BOARDSIZE]; 
    long long aQueenBitCol[MAX_BOARDSIZE]; 
    long long aQueenBitPosDiag[MAX_BOARDSIZE]; 
    long long aQueenBitNegDiag[MAX_BOARDSIZE]; 
    long long aStack[MAX_BOARDSIZE]; 
     
    register long long int *pnStack;
    register long long int pnStackPos = 0LLU;
    
    
    long long int board_minus = board_size - 1LL; 
    long long int mask = (1LL << board_size) - 1LL; 
    
    unsigned long long local_num_sols = 0ULL;
    unsigned long long tree_size = 0ULL;
    
    register unsigned long long lsb; 
    register unsigned long long bitfield; 

    
    register long long numrows = cutoff_depth;
    
    aStack[0] = -1LL; 
    
    pnStack = aStack; 
    
    aQueenBitRes[numrows] = subproblem->aQueenBitRes; 
    aQueenBitCol[numrows] = subproblem->aQueenBitCol; 
    aQueenBitPosDiag[numrows] = subproblem->aQueenBitPosDiag; 
    aQueenBitNegDiag[numrows] = subproblem->aQueenBitNegDiag; 
    

    bitfield = mask & ~(aQueenBitCol[numrows] | aQueenBitNegDiag[numrows] | aQueenBitPosDiag[numrows]);
    

    for (;;)
    {
    
        lsb = -((signed long long)bitfield) & bitfield; 
        
        if (0ULL == bitfield)
        {
           
            if(numrows <= cutoff_depth){ 
                //printf("\nEND OF THE SUBPROBLEM EXPLORATION! %d", numrows);
                break ;
            }

            bitfield = *--pnStack; /* get prev. bitfield from stack */
            
            --numrows;
            continue;
        }

        bitfield &= ~lsb; 
        
        aQueenBitRes[numrows] = lsb;
        
        if (numrows < board_minus) 
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


    printf("\nRESULTS: size %lld, Tree: %llu,  solutions: %llu\n", board_size,total_mcore_tree_size+initial_tree_size, total_mcore_num_sols*2);
    
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



