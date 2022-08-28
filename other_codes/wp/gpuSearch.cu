

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <cmath>
#include <omp.h>
#include <sys/time.h>

#define MAX_BOARDSIZE 32
#define __BLOCK_SIZE__ 128

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


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


void displayBitsLLU(unsigned long long value){

    unsigned long long  SHIFT = 8 * sizeof( unsigned long long ) - 1;
    unsigned long long MASK = (unsigned long long)1 << SHIFT ;// joga do menos para o mais significativo

    //cout<< setw(7) << value << " = ";

    for (unsigned long long c = 1; c <= SHIFT +1; c++){
        printf("%llu", value & MASK ? 1LLU: 0LLU);
        value <<= 1;
        if (c % 8 == 0)
            printf(" ");

    }

    printf("\n");

}


typedef struct subproblem{
    
    long long  aQueenBitRes[MAX_BOARDSIZE]; /* results */
    long long  aQueenBitCol[MAX_BOARDSIZE]; /* marks colummns which already have queens */
    long long  aQueenBitPosDiag[MAX_BOARDSIZE]; /* marks "positive diagonals" which already have queens */
    long long  aQueenBitNegDiag[MAX_BOARDSIZE]; /* marks "negative diagonals" which already have queens */
    long long  subproblem_stack[MAX_BOARDSIZE+2]; /* we use a stack instead of recursion */
  
    long long int pnStackPos;
    long long numrows; /* numrows redundant - could use stack */
    unsigned long long num_sols_sub;
  
} Subproblem;



unsigned long long partial_search(long long board_size, long long cutoff_depth, Subproblem *subproblem_pool)
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

                   // printf("\nSub: ");
                   // for(int i = 0; i<cutoff_depth;++i){
                   //      printf(" %d - ", (unsigned int)(log2(aQueenBitRes[i])+1));
                   // }
                    
                    printf("\n");
                 
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


__global__ void gpu_final_search(long long board_size, long long cutoff_depth, 
    unsigned long long num_subproblems, Subproblem* subproblems, unsigned long long *tree_size_d,
    unsigned long long *sols_d){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_subproblems) {


        //Bring this to the thread stack
       // long long* aQueenBitRes =     subproblems[idx].aQueenBitRes; 
       // long long* aQueenBitCol =     subproblems[idx].aQueenBitCol; 
       // long long* aQueenBitPosDiag = subproblems[idx].aQueenBitPosDiag; 
       // long long* aQueenBitNegDiag = subproblems[idx].aQueenBitNegDiag ; 
       // long long* aStack =           subproblems[idx].subproblem_stack ; 

        long long aQueenBitCol[32]; 
        long long aQueenBitPosDiag[32]; 
        long long aQueenBitNegDiag[32];  
        long long aStack[32]; 

        long long int pnStackPos = subproblems[idx].pnStackPos;

        long long *pnStack;

        long long int board_minus = board_size - 1LL; /* board size - 1 */
        long long int mask = (1LL << board_size) - 1LL; /* if board size is N, mask consists of N 1's */

        unsigned long long local_num_sols = 0ULL;

        unsigned long long lsb; 
        unsigned long long bitfield; 

        unsigned long long tree_size = 0ULL;

        long long numrows = cutoff_depth;


        for(int i = 0; i<cutoff_depth+2;++i){
            aQueenBitCol[i]  =     subproblems[idx].aQueenBitCol[i] ; 
            aQueenBitPosDiag[i]  = subproblems[idx].aQueenBitPosDiag[i] ; 
            aQueenBitNegDiag[i]  = subproblems[idx].aQueenBitNegDiag[i]  ; 
            aStack[i]  =           subproblems[idx].subproblem_stack[i]  ; 
        }
        

        pnStack = aStack + pnStackPos; /* stack pointer */
        bitfield = mask & ~(aQueenBitCol[numrows] | aQueenBitNegDiag[numrows] | aQueenBitPosDiag[numrows]);
                
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
        sols_d[idx] = local_num_sols;
        tree_size_d[idx] = tree_size;


    }//num_subproblems

}//kernel


unsigned long long mcore_final_search(long long board_size, long long cutoff_depth, Subproblem* subproblem, int index)
{

 
    long long* aQueenBitRes = subproblem->aQueenBitRes; 
    long long* aQueenBitCol = subproblem->aQueenBitCol; 
    long long* aQueenBitPosDiag = subproblem->aQueenBitPosDiag; 
    long long* aQueenBitNegDiag = subproblem->aQueenBitNegDiag ; 
    long long* aStack = subproblem->subproblem_stack ; 

    register long long *pnStack;
    
    long long int pnStackPos = subproblem->pnStackPos;

    long long int board_minus = board_size - 1LL; /* board size - 1 */
    long long int mask = (1LL << board_size) - 1LL; /* if board size is N, mask consists of N 1's */

    unsigned long long local_num_sols = 0ULL;

    register unsigned long long lsb; 
    register unsigned long long bitfield; 

   // printf("\nSubproblem: %d :\n\t ", index);
   //     for(int i = 0; i<cutoff_depth;++i){
   //         printf(" %d - ", (unsigned int)(log2(aQueenBitRes[i])+1));
   //     }
   // printf("\n");

    unsigned long long tree_size = 0ULL;


    register long long numrows = cutoff_depth;

    pnStack = aStack + pnStackPos; /* stack pointer */
    bitfield = mask & ~(aQueenBitCol[numrows] | aQueenBitNegDiag[numrows] | aQueenBitPosDiag[numrows]);
            
   // displayBitsLLU(bitfield);
    
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

           // printf("\n");
           //  for(int i = 0; i<board_size;++i){
           //       printf(" %d - ", (unsigned int)(log2(aQueenBitRes[i])+1));
           // }
            // printf("\n");

            //++g_numsolutions;
            ++local_num_sols;
            bitfield = *--pnStack;
            --numrows;
            continue;
        }
    }

    //returning the number of solutions
    subproblem->num_sols_sub = local_num_sols;

    return tree_size;
}



void call_mcore_search(long long board_size, long long cutoff_depth){

    
    unsigned long long num_sols_search = 0ULL;
    unsigned long long num_subproblems = 0ULL;
    Subproblem *subproblem_pool = (Subproblem*)(malloc(sizeof(Subproblem)* 10000000));

    unsigned long long tree_size = partial_search(board_size,cutoff_depth, subproblem_pool);
    num_subproblems = g_numsolutions;
    g_numsolutions = 0ULL;

    printf("\nTree: %llu -- Pool: %llu \n", tree_size, num_subproblems);

  
    printf("\n - Parallel search! \n");

    #pragma omp parallel for schedule(dynamic,128) default(none)\
    shared(num_subproblems,board_size, cutoff_depth, subproblem_pool)\
    reduction(+:tree_size,num_sols_search)
    for(int s = 0; s<num_subproblems; ++s){
        tree_size+=mcore_final_search(board_size, cutoff_depth, subproblem_pool+s, s);
        num_sols_search+=subproblem_pool[s].num_sols_sub;
    }


    if (num_sols_search != 0)
    {
        printf("PARALLEL SEARCH: size %lld, Tree: %llu,  solutions: %llu\n", board_size, tree_size, num_sols_search*2);
    }
    else
    {
        printf("No solutions found.\n");
    }
    printf("\n#######################################\n");
    free(subproblem_pool);

}////////////////////////////////////////////////



void call_gpu_search(long long board_size, long long cutoff_depth){

    
    double initial_time = rtclock();
    unsigned long long num_sols_search = 0ULL;
    unsigned long long n_explorers = 0ULL;
    Subproblem *subproblems_h = (Subproblem*)(malloc(sizeof(Subproblem)* (unsigned)10000000));

    unsigned long long gpu_tree_size = 0ULL;
    unsigned long long initial_tree_size = partial_search(board_size,cutoff_depth, subproblems_h);
    n_explorers = g_numsolutions;
   
    cudaFuncSetCacheConfig(gpu_final_search, cudaFuncCachePreferL1);
    
    printf("\nPartial serach: %llu -- Pool: %llu \n", initial_tree_size, n_explorers);

    unsigned long long int *vector_of_tree_size_h = (unsigned long long int*)malloc(sizeof(unsigned long long int)*n_explorers );
    unsigned long long int *sols_h = (unsigned long long int*)malloc(sizeof(unsigned long long int)*n_explorers );

    unsigned long long int *vector_of_tree_size_d;
    unsigned long long int *sols_d;
    Subproblem *subproblems_d;

    cudaMalloc((void**) &vector_of_tree_size_d,n_explorers*sizeof(unsigned long long int));
    cudaMalloc((void**) &sols_d,n_explorers*sizeof(unsigned long long int));
    cudaMalloc((void**) &subproblems_d,n_explorers*sizeof(Subproblem));

    cudaMemcpy(subproblems_d, subproblems_h, n_explorers * sizeof(Subproblem), cudaMemcpyHostToDevice);


    int num_blocks = ceil((double)n_explorers/__BLOCK_SIZE__);

    printf("\n### Regular BP-DFS search. ###\n");

    gpu_final_search<<< num_blocks,__BLOCK_SIZE__>>>(board_size,cutoff_depth,n_explorers,subproblems_d,vector_of_tree_size_d,sols_d);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );


    cudaMemcpy(vector_of_tree_size_h,vector_of_tree_size_d,n_explorers*sizeof(unsigned long long int),cudaMemcpyDeviceToHost);
    cudaMemcpy(sols_h,sols_d,n_explorers*sizeof(unsigned long long int),cudaMemcpyDeviceToHost);

    cudaFree(vector_of_tree_size_d);
    cudaFree(sols_d);
    cudaFree(subproblems_d);


    double final_time = rtclock();
    for(int i = 0; i<n_explorers;++i){
        if(sols_h[i]>0)
            num_sols_search += sols_h[i];
        if(vector_of_tree_size_h[i]>0)
            gpu_tree_size +=vector_of_tree_size_h[i];
    }


    printf("\nGPU Tree size: %llu\nTotal tree size: %llu\nNumber of solutions found: %llu.\n", gpu_tree_size,(initial_tree_size+gpu_tree_size),num_sols_search*2LLU );
    printf("\nElapsed total: %.3f\n", (final_time-initial_time));


    free(sols_h);
    free(vector_of_tree_size_h);
}////////////////////////////////////////////////



/* main routine for N Queens program.*/
int main(int argc, char** argv)
{
    
    int boardsize;

    boardsize = atoi(argv[1]);

    call_gpu_search(boardsize, (long long)(atoi(argv[2])));
    //call_mcore_search(boardsize, (long long)(atoi(argv[2])));
    
    return 0;
}

