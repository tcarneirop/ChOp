
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <cmath>
#include <omp.h>
#include <sys/time.h>
#include <mpi.h>
#include <climits>

/// this is used to check if the solution produced is correct
unsigned long long check_sols_number[] = {0,	0,	0,	2,	10,	4,	40,	92,	352,	724,	2680,	14200,	73712,	
365596,	2279184,	14772512,	95815104,	666090624,	4968057848,	39029188884,314666222712,2691008701644,24233937684440,227514171973736 };


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
        bitfield = 0ULL;
        if (0LL == i)
        {
            long long int half = board_size>>1LL;
            bitfield = (1LL << half) - 1LL;
            pnStack = aStack + 1LL; /* stack pointer */
            
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
            pnStack = aStack + 1LL; /* stack pointer */
            
            pnStackPos++;

            *pnStack++ = 0LL; /* we're done w/ this row -- only 1 element & we've done it */
            bitfield = (bitfield - 1ULL) >> 1ULL; /* bitfield -1 is all 1's to the left of the single 1 */
        }

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



unsigned long long queens_get_rank_load(int mpi_rank, int num_ranks, unsigned long long num_subproblems){
    unsigned long long rank_load = num_subproblems/num_ranks;
    return (mpi_rank == (num_ranks-1) ? rank_load + (num_subproblems % num_ranks): rank_load);
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
    
    long long int board_minus = board_size - 1LL; /* board size - 1 */
    long long int mask = (1LL << board_size) - 1LL; /* if board size is N, mask consists of N 1's */

    unsigned long long local_num_sols = 0ULL;
    unsigned long long tree_size = 0ULL;
    
    register unsigned long long lsb; 
    register unsigned long long bitfield; 

    
    register long long numrows = cutoff_depth;

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


void call_MPI_mcore_search(long long board_size, long long cutoff_depth, int mpi_rank, int num_ranks){
    
    unsigned long long rank_num_sols = 0ULL;
    unsigned long long rank_tree_size = 0ULL;

    unsigned long long global_num_sols = 0ULL;
    unsigned long long global_tree_size = 0ULL;
    double             global_exec_time = 0.;

    double rank_initial_time = rtclock();


    Subproblem *subproblem_pool = (Subproblem*)(malloc(sizeof(Subproblem)*(unsigned)1000000));
    
    g_numsolutions = 0ULL;
    
    unsigned long long initial_tree_size = partial_search_64(board_size,cutoff_depth, subproblem_pool);
    
    unsigned long long num_subproblems = g_numsolutions;

    unsigned long long num_sols_search = 0ULL;


    unsigned long long rank_load = queens_get_rank_load(mpi_rank,num_ranks, num_subproblems);
    subproblem_pool = subproblem_pool + num_subproblems/num_ranks*mpi_rank;

    unsigned long long mcore_tree_size[rank_load];
    unsigned long long mcore_num_sols[rank_load];

    for(unsigned long long i = 0; i<rank_load;++i){
        mcore_num_sols[i] = 0ULL;
        mcore_tree_size[i] = 0ULL;
    }

    if(mpi_rank == 0){
        printf("\n### Queens size: %lld, Num subproblems: %llu, Initial depth: %lld - rank_load: %llu - num_threads: %d", board_size, num_subproblems,cutoff_depth,rank_load, omp_get_max_threads());
    }


    #pragma omp parallel for schedule(runtime) default(none) shared(rank_load,board_size,mcore_tree_size,mcore_num_sols, cutoff_depth, subproblem_pool)
    for(int s = 0; s<rank_load; ++s){
        mcore_final_search(board_size, cutoff_depth, subproblem_pool+s, s,mcore_tree_size,mcore_num_sols);
    }

    for(int s = 0; s<rank_load;++s){
        rank_tree_size+=mcore_tree_size[s];
        rank_num_sols+=mcore_num_sols[s];
    }


       //printf("\n Rank %d Tree size: %llu", mpi_rank,rank_tree_size);

    MPI_Allreduce(&rank_tree_size, &global_tree_size, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&rank_num_sols, &global_num_sols, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        
    double rank_final_time = rtclock();
    double rank_total_time = rank_final_time - rank_initial_time;

    MPI_Allreduce(&rank_total_time, &global_exec_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);


    if(mpi_rank == 0){
        global_num_sols*=2;
        printf("\nFinal Tree size: %llu\nNumber of solutions found: %llu\n", global_tree_size+initial_tree_size,global_num_sols);
        printf("\nElapsed total: %.3f\n", global_exec_time);
    }


    #ifdef CHECKSOL
        if(mpi_rank == 0){
            if(global_num_sols == check_sols_number[board_size-1])
                printf("\n####### SUCCESS - CORRECT NUMBER OF SOLS. FOR SIZE %lld\n", board_size);
            else
                printf("########## ERROR -- INCORRECT NUMBER FOS SOLS. FOR SIZE %lld\n", board_size);
                //exit(1);
        }
    #endif


    #ifdef RANKLOADS

    unsigned long long rank_loads[num_ranks];
    double rank_exec_times[num_ranks];
    
    MPI_Gather(&rank_tree_size, 1, MPI_UNSIGNED_LONG_LONG, rank_loads, 1, MPI_UNSIGNED_LONG_LONG, 0,MPI_COMM_WORLD);
    MPI_Gather(&rank_total_time, 1, MPI_DOUBLE, rank_exec_times, 1, MPI_DOUBLE, 0,MPI_COMM_WORLD);
    
    if( mpi_rank == 0 ){
        unsigned long long biggest = 0;
        
        int biggest_load;
        int smallest_load;

        unsigned long long smallest = ULLONG_MAX;
      
        
        printf("\n\n########## Per-rank Load Report: <rank> <tree_size> <exec_time> \n");
        for(int rank = 0; rank<num_ranks;++rank){
            
            if (rank_loads[rank] < smallest) {smallest = rank_loads[rank]; smallest_load = rank;}
            if (rank_loads[rank] > biggest)  {biggest = rank_loads[rank]; biggest_load = rank;} 


            printf("\tRank %d - Local tree: %llu - %.3f\n", rank, rank_loads[rank], rank_exec_times[rank] );

        }

        printf("\n\tBiggest local tree: %llu -  %.3f ", rank_loads[biggest_load], rank_exec_times[biggest_load] );
        printf("\n\tSmallest local tree: %llu - %.3f ", rank_loads[smallest_load], rank_exec_times[smallest_load] );
        printf("\n\tBiggest/Smallest: %.3f\n", (double)rank_loads[biggest_load]/(double)rank_loads[smallest_load]);

    }
    #endif



}////////////////////////////////////////////////



/* main routine for N Queens program.*/
int main(int argc, char** argv)
{

    
    int size;
    int initialDepth;
    int mpi_rank;
    int num_ranks;

   
    if (argc != 3) {
        printf("Usage: %s <size> <initial depth>\n", argv[0]);
        return 1;
    }

    size = atoi(argv[1]);
    initialDepth = atoi(argv[2]);
    
    MPI_Init(&argc, &argv);
    
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    
    if(mpi_rank == 0) printf("\nNumber of MPI Ranks: %d", num_ranks);
    
    call_MPI_mcore_search(size,initialDepth,mpi_rank,num_ranks);


    MPI_Finalize();


    return 0;
}



