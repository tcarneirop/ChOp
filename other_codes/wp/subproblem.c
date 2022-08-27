

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <cmath>
#include <omp.h>


#define MAX_BOARDSIZE 64
typedef unsigned long long SOLUTIONTYPE;

#define MIN_BOARDSIZE 2

SOLUTIONTYPE g_numsolutions = 0ULL;


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
    long long  subproblem_stack[MAX_BOARDSIZE + 2]; /* we use a stack instead of recursion */
  
    long long int pnStackPos;
    long long numrows; /* numrows redundant - could use stack */
    unsigned long long num_sols_sub;
  
} Subproblem;



unsigned long long partial_search(long long board_size, long long cutoff_depth, Subproblem *subproblem_pool)
{

    long long permutation[MAX_BOARDSIZE]; 
    long long aQueenBitRes[MAX_BOARDSIZE]; /* results */
    long long aQueenBitCol[MAX_BOARDSIZE]; /* marks colummns which already have queens */
    long long aQueenBitPosDiag[MAX_BOARDSIZE]; /* marks "positive diagonals" which already have queens */
    long long aQueenBitNegDiag[MAX_BOARDSIZE]; /* marks "negative diagonals" which already have queens */
    long long aStack[MAX_BOARDSIZE + 2]; /* we use a stack instead of recursion */
    long long subproblem_stack[MAX_BOARDSIZE + 2];
    
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

unsigned long long parallel_search(long long board_size, long long cutoff_depth, Subproblem* subproblem, int index)
{

 
    long long* aQueenBitRes = subproblem->aQueenBitRes; 
    long long* aQueenBitCol = subproblem->aQueenBitCol; 
    long long* aQueenBitPosDiag = subproblem->aQueenBitPosDiag; 
    long long* aQueenBitNegDiag = subproblem->aQueenBitNegDiag ; 
    long long* aStack = subproblem->subproblem_stack ; 

    register long long *pnStack;
    
    long long int pnStackPos = subproblem->pnStackPos;

    long long int odd = board_size & 1LL; /* 0 if board_size even, 1 if odd */
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





void call_parallel_search(long long board_size, long long cutoff_depth){

    
    unsigned long long num_sols_search = 0ULL;
    unsigned long long num_subproblems = 0ULL;
    Subproblem *subproblem_pool = (Subproblem*)(malloc(sizeof(Subproblem)* 1000000));

    unsigned long long tree_size = partial_search(board_size,cutoff_depth, subproblem_pool);
    num_subproblems = g_numsolutions;
    g_numsolutions = 0ULL;

    printf("\nTree: %llu -- Pool: %llu \n", tree_size, num_subproblems);

    //exit(1);
    printf("\n - Parallel search! \n");
    for(int s = 0; s<num_subproblems; ++s){
        tree_size+=parallel_search(board_size, cutoff_depth, subproblem_pool+s, s);
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


void Nqueen(long long int board_size)
{
    long long int aQueenBitRes[MAX_BOARDSIZE]; /* results */
    long long int aQueenBitCol[MAX_BOARDSIZE]; /* marks colummns which already have queens */
    long long int aQueenBitPosDiag[MAX_BOARDSIZE]; /* marks "positive diagonals" which already have queens */
    long long int aQueenBitNegDiag[MAX_BOARDSIZE]; /* marks "negative diagonals" which already have queens */
    long long int aStack[MAX_BOARDSIZE + 2]; /* we use a stack instead of recursion */
    
    int pnstackPos = 0;

    register long long int* pnStack;

    register long long int numrows = 0LL; /* numrows redundant - could use stack */
    register unsigned long long int lsb; /* least significant bit */
    register unsigned long long int bitfield; /* bits which are set mark possible positions for a queen */
    long long int i;
    long long int odd = board_size & 1LL; /* 0 if board_size even, 1 if odd */
    long long int board_minus = board_size - 1LL; /* board size - 1 */
    //Change here for the pool
    //long long int board_minus = 45LL; /* board size - 1 */
    long long int mask = (1LL << board_size) - 1LL; /* if board size is N, mask consists of N 1's */

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
            printf("Boardsize: %lld\n", board_size);
            displayBitsLLU(board_size);

            printf("\nhalf: %lld\n", half);
            displayBitsLLU(half);

            /* fill in rightmost 1's in bitfield for half of board_size
               If board_size is 7, half of that is 3 (we're discarding the remainder)
               and bitfield will be set to 111 in binary. */

            bitfield = (1LL << half) - 1LL;
            pnStack = aStack + 1LL; /* stack pointer */
           ++pnstackPos;

            printf("\nBitfield inicial: ");
            displayBitsLLU(bitfield);

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
            *pnStack++ = 0LL; /* we're done w/ this row -- only 1 element & we've done it */
            ++pnstackPos;
            bitfield = (bitfield - 1ULL) >> 1ULL; /* bitfield -1 is all 1's to the left of the single 1 */
            

           // displayBitsLLU(bitfield);
        }

        /* this is the critical loop */
        for (;;)
        {
            /* could use
               lsb = bitfield ^ (bitfield & (bitfield -1));
               to get first (least sig) "1" bit, but that's slower. */
            lsb = -((signed long long int)bitfield) & bitfield; /* this assumes a 2's complement architecture */
            
            printf("\nLSB: %d \n", (unsigned)log2(lsb));
            displayBitsLLU(bitfield);
            

            if (0ULL == bitfield)
            {
                
                bitfield = *--pnStack; /* get prev. bitfield from stack */
                --pnstackPos;

                if (pnStack == aStack) { /* if sentinel hit.... */
                    
                    printf("\nNum rows before breaking: %lld \nBreak################################################", numrows);
                   
                    break ;
                }

                //printf("\nBacktracking 0ull- numrows - %lld, stack position: %d\n", numrows,pnstackPos);
                --numrows;
                continue;
            }

            bitfield &= ~lsb; /* toggle off this bit so we don't try it again */

            printf("\n not lsb %llu and bitfield: \n", lsb);
            displayBitsLLU(bitfield);

    
            aQueenBitRes[numrows] = lsb; /* save the result */
            printf("numrows: %lld\n", numrows);
            
            if (numrows < board_minus) /* we still have more rows to process? */
            {
                long long int n = numrows++;
                aQueenBitCol[numrows] = aQueenBitCol[n] | lsb;
                aQueenBitNegDiag[numrows] = (aQueenBitNegDiag[n] | lsb) >> 1LL;
                aQueenBitPosDiag[numrows] = (aQueenBitPosDiag[n] | lsb) << 1LL;
                *pnStack++ = bitfield;
                ++pnstackPos;

                printf("\nbitfield interno: \n");
                displayBitsLLU(bitfield);
                
                /* We can't consider positions for the queen which are in the same
                   column, same positive diagonal, or same negative diagonal as another
                   queen already on the board. */
                bitfield = mask & ~(aQueenBitCol[numrows] | aQueenBitNegDiag[numrows] | aQueenBitPosDiag[numrows]);
                
                printf("\nbitfield interno arvore %llu: \n", tree_size+1);
                displayBitsLLU(bitfield);
                

                ++tree_size;

                continue;
            }
            else
            {
                /* We have no more rows to process; we found a solution. */
                /* Comment out the call to printtable in order to print the solutions as board position*/
                //printtable(board_size, aQueenBitRes, g_numsolutions + 1,tree_size);
                   for(int i = 0; i<board_size;++i){
                       printf(" %d - ", (unsigned int)(log2(aQueenBitRes[i])+1));
                   }
                 printf("\n");

                printf("\nBacktracking 0ull- numrows - %lld, stack position: %d\n", numrows,pnstackPos);
                
                ++g_numsolutions;
                bitfield = *--pnStack;
                --pnstackPos;
                
                //printf("\nBacktracking sol: \n");
                --numrows;
                // exit(1);
                continue;
            }
        }

    }



    /* multiply solutions by two, to count mirror images */
    printf("\nTree: %llu -- BPDFS tree: %llu \n", tree_size, tree_size*2 );
    g_numsolutions *= 2;
}


void completeEnumeration(long long int board_size)
{
    long long int aQueenBitRes[MAX_BOARDSIZE]; /* results */
    long long int aQueenBitCol[MAX_BOARDSIZE]; /* marks colummns which already have queens */
    long long int aStack[MAX_BOARDSIZE + 2]; /* we use a stack instead of recursion */
    
    int pnstackPos = 0;

    register long long int* pnStack;

    register long long int numrows = 0LL; /* numrows redundant - could use stack */
    register unsigned long long int lsb; /* least significant bit */
    register unsigned long long int bitfield; /* bits which are set mark possible positions for a queen */
    long long int i;
    long long int odd = board_size & 1LL; /* 0 if board_size even, 1 if odd */
    long long int board_minus = board_size - 1LL; /* board size - 1 */
    //Change here for the pool
    //long long int board_minus = 45LL; /* board size - 1 */
    long long int mask = (1LL << board_size) - 1LL; /* if board size is N, mask consists of N 1's */

    unsigned long long tree_size = 0ULL;
    /* Initialize stack */
    aStack[0] = -1LL; /* set sentinel -- signifies end of stack */

    for (i = 0; i < (1); ++i)
    {
        /* We don't have to optimize this part; it ain't the
           critical loop */
        bitfield = 0ULL;
       // if (0LL == i)
       // {
            /* Handle half of the board, except the middle
               column. So if the board is 5 x 5, the first
               row will be: 00011, since we're not worrying
               about placing a queen in the center column (yet).
            */
            long long int half = board_size; /* divide by two */
            /* fill in rightmost 1's in bitfield for half of board_size
               If board_size is 7, half of that is 3 (we're discarding the remainder)
               and bitfield will be set to 111 in binary. */
            bitfield = (1LL << half) - 1LL;
            pnStack = aStack + 1LL; /* stack pointer */

            printf("\nBitfield inicial: ");
            displayBitsLLU(bitfield);

            aQueenBitRes[0] = 0LL;
            aQueenBitCol[0] = 0LL;
 
        //}
    

        /* this is the critical loop */
        for (;;)
        {
            /* could use
               lsb = bitfield ^ (bitfield & (bitfield -1));
               to get first (least sig) "1" bit, but that's slower. */
            lsb = -((signed long long int)bitfield) & bitfield; /* this assumes a 2's complement architecture */
            
            //printf("\nLSB: %d \n", (unsigned)log2(lsb));
           // displayBitsLLU(bitfield);

            if (0ULL == bitfield)
            {
                bitfield = *--pnStack; /* get prev. bitfield from stack */
                
                if (pnStack == aStack) { /* if sentinel hit.... */
            
                    break ; //end of the search
                }

               // printf("\nBacktracking 0ull- numrows - %lld", numrows-1);
                --numrows;
                continue;
            }

            bitfield &= ~lsb; /* toggle off this bit so we don't try it again */

            //printf("\n%llu and bitfield: \n", lsb);
            //displayBitsLLU(bitfield);
            //printf("numrows: %lld\n", numrows);


            aQueenBitRes[numrows] = lsb; /* save the result */

            if (numrows < board_minus) /* we still have more rows to process? */
            {
                long long int n = numrows++;
                aQueenBitCol[numrows] = aQueenBitCol[n] | lsb;
                //aQueenBitNegDiag[numrows] = (aQueenBitNegDiag[n] | lsb) >> 1LL;
                //aQueenBitPosDiag[numrows] = (aQueenBitPosDiag[n] | lsb) << 1LL;
                *pnStack++ = bitfield;

               // printf("\nbitfield interno: \n");
                //displayBitsLLU(bitfield);

                /* We can't consider positions for the queen which are in the same
                   column, same positive diagonal, or same negative diagonal as another
                   queen already on the board. */
                bitfield = mask & ~(aQueenBitCol[numrows]);
                
               // printf("\nbitfield interno arvore %llu: \n", tree_size+1);
               // displayBitsLLU(bitfield);
                ++tree_size;
                continue;
            }
            else
            {
                //printtable(board_size, aQueenBitRes, g_numsolutions + 1,tree_size);
                 //  for(int i = 0; i<board_size;++i){
                 //      printf(" %d - ", (unsigned int)(log2(aQueenBitRes[i])+1));
                 // }
                //printf("\n");

                ++g_numsolutions;
                bitfield = *--pnStack;

                //printf("\nBacktracking sol: \n");
                --numrows;
                //exit(1);
                continue;
            }
        }

    }


    /* multiply solutions by two, to count mirror images */
    printf("\nTree: %llu \n", tree_size );
    //g_numsolutions = 2;
}

/* Print the results at the end of the run */
void printResults(time_t* pt1, time_t* pt2)
{
    double secs;
    int hours , mins, intsecs;

    printf("End: \t%s", ctime(pt2));
    secs = difftime(*pt2, *pt1);
    intsecs = (int)secs;
    printf("Calculations took %d second%s.\n", intsecs, (intsecs == 1 ? "" : "s"));

    /* Print hours, minutes, seconds */
    hours = intsecs/3600;
    intsecs -= hours * 3600;
    mins = intsecs/60;
    intsecs -= mins * 60;
    if (hours > 0 || mins > 0)
    {
        printf("Equals ");
        if (hours > 0)
        {
            printf("%d hour%s, ", hours, (hours == 1) ? "" : "s");
        }
        if (mins > 0)
        {
            printf("%d minute%s and ", mins, (mins == 1) ? "" : "s");
        }
        printf("%d second%s.\n", intsecs, (intsecs == 1 ? "" : "s"));

    }
}

void call_serial_queens(){

}

void call_complete_enumeration(){

}



/* main routine for N Queens program.*/
int main(int argc, char** argv)
{
    time_t t1, t2;
    int boardsize;

    boardsize = atoi(argv[1]);

    time(&t1);
    printf("Start: \t %s", ctime(&t1));


    
    //partial_search(boardsize, (long long)(atoi(argv[2])));

    call_parallel_search(boardsize, (long long)(atoi(argv[2])));

    // time(&t2);

    // printResults(&t1, &t2);
   // g_numsolutions = 0ULL;
   //Nqueen(boardsize);
   // completeEnumeration(boardsize);
    // if (g_numsolutions != 0)
    // {
    //     printf("For board size %d, %llu solution %s found.\n", boardsize, g_numsolutions, (g_numsolutions == 1 ? "" : "s"));
    // }
    // else
   // {
   //      printf("No solutions found.\n");
   // }

    return 0;
}

