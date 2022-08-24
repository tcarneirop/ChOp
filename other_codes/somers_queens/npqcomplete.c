/*  Jeff Somers
 *
 *  Copyright (c) 2002
 *
 *  jsomers@alumni.williams.edu
 *  or
 *  allagash98@yahoo.com
 *
 *  April, 2002
 *
 *  Program:  nq
 *
 *  Program to find number of solutions to the N queens problem.
 *  This program assumes a twos complement architecture.
 *
 *  For example, you can arrange 4 queens on 4 x 4 chess so that
 *  none of the queens can attack each other:
 *
 *  Two solutions:
 *     _ Q _ _        _ _ Q _
 *     _ _ _ Q        Q _ _ _
 *     Q _ _ _        _ _ _ Q
 *     _ _ Q _    and _ Q _ _
 *
 *  Note that these are separate solutions, even though they
 *  are mirror images of each other.
 *
 *  Likewise, a 8 x 8 chess board has 92 solutions to the 8 queens
 *  problem.
 *
 *  Command Line Usage:
 *
 *          nq N
 *
 *       where N is the size of the N x N board.  For example,
 *       nq 4 will find the 4 queen solution for the 4 x 4 chess
 *       board.
 *
 *  By default, this program will only print the number of solutions,
 *  not board arrangements which are the solutions.  To print the
 *  boards, uncomment the call to printtable in the Nqueen function.
 *  Note that printing the board arrangements slows down the program
 *  quite a bit, unless you pipe the output to a text file:
 *
 *  nq 10 > output.txt
 *
 *
 *  The number of solutions for the N queens problems are known for
 *  boards up to 23 x 23.  With this program, I've calculated the
 *  results for boards up to 21 x 21, and that took over a week on
 *  an 800 MHz PC.  The algorithm is approximated O(n!) (i.e. slow),
 *  and calculating the results for a 22 x 22 board will take about 8.5
 *  times the amount of time for the 21 x 21 board, or over 8 1/2 weeks.
 *  Even with a 10 GHz machine, calculating the results for a 23 x 23
 *  board would take over a month.  Of course, setting up a cluster of
 *  machines (or a distributed client) would do the work in less time.
 *
 *  (from Sloane's On-Line Encyclopedia of Integer Sequences,
 *   Sequence A000170
 *   http://www.research.att.com/cgi-bin/access.cgi/as/njas/sequences/eisA.cgi?Anum=000170
 *   )
 *
 *   Board Size:       Number of Solutions to          Time to calculate
 *   (length of one        N queens problem:              on 800MHz PC
 *    side of N x N                                    (Hours:Mins:Secs)
 *    chessboard)
 *
 *     1                                  1                    n/a
 *     2                                  0                   < 0 seconds
 *     3                                  0                   < 0 seconds
 *     4                                  2                   < 0 seconds
 *     5                                 10                   < 0 seconds
 *     6                                  4                   < 0 seconds
 *     7                                 40                   < 0 seconds
 *     8                                 92                   < 0 seconds
 *     9                                352                   < 0 seconds
 *    10                                724                   < 0 seconds
 *    11                               2680                   < 0 seconds
 *    12                              14200                   < 0 seconds
 *    13                              73712                   < 0 seconds
 *    14                             365596                  00:00:01
 *    15                            2279184                  00:00:04
 *    16                           14772512                  00:00:23
 *    17                           95815104                  00:02:38
 *    18                          666090624                  00:19:26
 *    19                         4968057848                  02:31:24
 *    20                        39029188884                  20:35:06
 *    21                       314666222712                 174:53:45
 *    22                      2691008701644                     ?
 *    23                     24233937684440                     ?
 *    24                                  ?                     ?
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <cmath>



/*
   Notes on MAX_BOARDSIZE:

   A 32 bit unsigned long is sufficient to hold the results for an 18 x 18
   board (666090624 solutions) but not for a 19 x 19 board (4968057848 solutions).

   In Win32, I use a 64 bit variable to hold the results, and merely set the
   MAX_BOARDSIZE to 21 because that's the largest board for which I've
   calculated a result.

   Note: a 20x20 board will take over 20 hours to run on a Pentium III 800MHz,
   while a 21x21 board will take over a week to run on the same PC.

   On Unix, you could probably change the type of g_numsolutions from unsigned long
   to unsigned long long, or change the code to use two 32 bit ints to store the
   results for board sizes 19 x 19 and up.
*/

//#ifdef WIN32

//#define MAX_BOARDSIZE 24
//typedef unsigned __int64 SOLUTIONTYPE;

//#else

#define MAX_BOARDSIZE 64
typedef unsigned long long SOLUTIONTYPE;

//#endif

#define MIN_BOARDSIZE 2

SOLUTIONTYPE g_numsolutions = 0ULL;


/* Print a chess table with queens positioned for a solution */
/* This is not a critical path function & I didn't try to optimize it. */
void printtable(long long int boardsize, long long int* aQueenBitRes, SOLUTIONTYPE numSolution, unsigned long long int partial_tree)
{
    long long int i, j, k, row, l=0;
    long long int permutation[boardsize];

    /*  We only calculated half the solutions, because we can derive
        the other half by reflecting the solution across the "Y axis". */
    for (k = 0LL; k < 2LL; ++k)
    {

        printf("*** Solution #: %lld ***\n", 2LL * numSolution + k - 1LL);
        for ( i = 0LL; i < boardsize; i++)
        {
            unsigned long long int bitf;
            /*
               Get the column that was set (i.e. find the
               first, least significant, bit set).
               If aQueenBitRes[i] = 011010b, then
               bitf = 000010b
            */
            bitf = aQueenBitRes[i];

            row = bitf ^ (bitf & (bitf - 1ULL)); /* get least significant bit */
            for ( j = 0LL; j < boardsize; j++)
            {
                // keep shifting row over to the right until we find the one '1' in
                //   the binary representation.  There will only be one '1'. 
                if (0LL == k && ((row >> j) & 1LL))
                {
                    //printf("Q %lld",j);
                    permutation[i] = j;
                }
                else if (1LL == k && (row & (1LL << (boardsize - j - 1LL)))) /* this is the board reflected across the "Y axis" */
                {
                    //printf("Q %lld",j);
                    permutation[i] = j;
                }
                else
                {
                    //printf(". ");
                }
            }
            //printf("\n");

         }
        printf("\n");
        for(l = 0; l<boardsize;++l)
            printf(" %lld - ", permutation[l]+1);
        printf("\n\tPartial tree: %lld", partial_tree );

    }
}


/* The function which calculates the N queen solutions.
   We calculate one-half the solutions, then flip the results over
   the "Y axis" of the board.  Every solution can be reflected that
   way to generate another unique solution (assuming the board size
   isn't 1 x 1).  That's because a solution cannot be symmetrical
   across the Y-axis (because you can't have two queens in the same
   horizontal row).  A solution also cannot consist of queens
   down the middle column of a board with an odd number of columns,
   since you can't have two queens in the same vertical row.

   This is a backtracking algorithm.  We place a queen in the top
   row, then note the column and diagonals it occupies.  We then
   place a queen in the next row down, taking care not to place it
   in the same column or diagonal.  We then update the occupied
   columns & diagonals & move on to the next row.  If no position
   is open in the next row, we back track to the previous row & move
   the queen over to the next available spot in its row & the process
   starts over again.
*/



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
    long long  aStack[MAX_BOARDSIZE + 2]; /* we use a stack instead of recursion */
    long long  permutation[MAX_BOARDSIZE]; /* we use a stack instead of recursion */

    long long numrows; /* numrows redundant - could use stack */
    unsigned long long lsb; /* least significant bit */
    unsigned long long bitfield; /* bits which are set mark possible positions for a queen */
    long long odd; /* 0 if board_size even, 1 if odd */
    long long board_minus;
    long long mask; /* if board size is N, mask consists of N 1's */
   
} Subproblem;



unsigned long long partial_search(long long board_size, long long cutoff_depth, Subproblem *subproblem_pool)
{

    long long permutation[MAX_BOARDSIZE]; 
    long long aQueenBitRes[MAX_BOARDSIZE]; /* results */
    long long aQueenBitCol[MAX_BOARDSIZE]; /* marks colummns which already have queens */
    long long aQueenBitPosDiag[MAX_BOARDSIZE]; /* marks "positive diagonals" which already have queens */
    long long aQueenBitNegDiag[MAX_BOARDSIZE]; /* marks "negative diagonals" which already have queens */
    long long aStack[MAX_BOARDSIZE + 2]; /* we use a stack instead of recursion */
    register long long * pnStack;


    register long long numrows = 0LL; /* numrows redundant - could use stack */
    register unsigned long long lsb; /* least significant bit */
    register unsigned long long bitfield; /* bits which are set mark possible positions for a queen */
    long long i;
    long long odd = board_size & 1LL; /* 0 if board_size even, 1 if odd */
    

    //cutoff here
    long long board_minus = cutoff_depth-1; /* board size - 1 */


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
            bitfield = (bitfield - 1ULL) >> 1ULL; /* bitfield -1 is all 1's to the left of the single 1 */
        }

        /* this is the critical loop */
        for (;;)
        {
            /* could use
               lsb = bitfield ^ (bitfield & (bitfield -1));
               to get first (least sig) "1" bit, but that's slower. */
            lsb = -((signed long long)bitfield) & bitfield; /* this assumes a 2's complement architecture */
            
            if (0ULL == bitfield)
            {
                
                bitfield = *--pnStack; /* get prev. bitfield from stack */

                ///
                /// if(pnStack=aStack[cutoff_depth-1])
                ///
                if (pnStack == aStack) { /* if sentinel hit.... */
                    break ;
                }
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
                /* We can't consider positions for the queen which are in the same
                   column, same positive diagonal, or same negative diagonal as another
                   queen already on the board. */
                bitfield = mask & ~(aQueenBitCol[numrows] | aQueenBitNegDiag[numrows] | aQueenBitPosDiag[numrows]);
                ++tree_size;
                continue;
            }
            else
            {
                /* We have no more rows to process; we found a solution. */
                /* Comment out the call to printtable in order to print the solutions as board position*/
                //printtable(board_size, aQueenBitRes, g_numsolutions + 1,tree_size);

                printf("\nSub: ");
                for(int i = 0; i<cutoff_depth;++i){
                     printf(" %d - ", (unsigned int)(log2(aQueenBitRes[i])+1));
                 }
                printf("\n");


                memcpy(subproblem_pool[g_numsolutions].aQueenBitRes, aQueenBitRes, sizeof(long long)*MAX_BOARDSIZE);
                memcpy(subproblem_pool[g_numsolutions].aQueenBitCol, aQueenBitCol, sizeof(long long)*MAX_BOARDSIZE);
                memcpy(subproblem_pool[g_numsolutions].aQueenBitPosDiag, aQueenBitPosDiag, sizeof(long long)*MAX_BOARDSIZE);
                memcpy(subproblem_pool[g_numsolutions].aStack, aStack, sizeof(long long)*(MAX_BOARDSIZE+2));
                memcpy(subproblem_pool[g_numsolutions].permutation, permutation, sizeof(long long)*MAX_BOARDSIZE );
                subproblem_pool[g_numsolutions].lsb = lsb;
                subproblem_pool[g_numsolutions].odd = odd;
                //subproblem_pool[g_numsolutions].board_minus = board_minus;
                subproblem_pool[g_numsolutions].mask = mask;
                subproblem_pool[g_numsolutions].numrows = numrows;  


                bitfield = *--pnStack;
                --numrows;

                subproblem_pool[g_numsolutions].bitfield = bitfield;

                ++g_numsolutions;


                //exit(1);
                continue;
            }
        }
    }

    return tree_size;

}

unsigned long long parallel_search(long long board_size, long long cutoff_depth, Subproblem* subproblem, int index)
{

    long long* permutation = subproblem->permutation; 
    long long* aQueenBitRes = subproblem->aQueenBitRes; 
    long long* aQueenBitCol = subproblem->aQueenBitCol; 
    long long* aQueenBitPosDiag = subproblem->aQueenBitPosDiag; 
    long long* aQueenBitNegDiag = subproblem->aQueenBitNegDiag ; 
    long long* aStack = subproblem->aStack ; 

    register long long * pnStack; /////HOW TO START THIS?
    //I believe that we need to point to the right place and update the numrows according to the cutoff. Dont knw yet..


    register long long numrows = subproblem->numrows;
    register unsigned long long lsb = subproblem->lsb; 
    register unsigned long long bitfield = subproblem->bitfield; 
    long long i;
    long long odd = board_size & 1LL; /* 0 if board_size even, 1 if odd */
    

    printf("\nSubproblem: %d :\n\t ", index);
        for(int i = 0; i<cutoff_depth;++i){
            printf(" %d - ", (unsigned int)(log2(aQueenBitRes[i])+1));
        }
    printf("\n");

    //cutoff here
    //dont know how to start... should i also modify the sentinell
    long long board_minus = board_size - 1LL; /* board size - 1 */

    long long mask = subproblem->mask; 

    unsigned long long tree_size = 0ULL;

    pnStack = aStack + cutoff_depth+1; /* stack pointer */
    numrows = cutoff_depth-1;

    // /* Initialize stack */

        /* this is the critical loop */
        for (;;)
        {
            /* could use
               lsb = bitfield ^ (bitfield & (bitfield -1));
               to get first (least sig) "1" bit, but that's slower. */
            lsb = -((signed long long)bitfield) & bitfield; /* this assumes a 2's complement architecture */
            if (0ULL == bitfield)
            {
                printf("ALORE!");
                bitfield = *--pnStack; /* get prev. bitfield from stack */
                if (*pnStack == aStack[cutoff_depth-1]) { /* if sentinel hit.... */
                    printf("BREAKING!");
                    break ;
                }
                printf("Backtracking!");
                --numrows;
                continue;
            }

            bitfield &= ~lsb; /* toggle off this bit so we don't try it again */

            aQueenBitRes[numrows] = lsb; /* save the result */
            if (numrows < board_minus) /* we still have more rows to process? */
            {
                printf("NOVOS!");
                long long n = numrows++;
                aQueenBitCol[numrows] = aQueenBitCol[n] | lsb;
                aQueenBitNegDiag[numrows] = (aQueenBitNegDiag[n] | lsb) >> 1LL;
                aQueenBitPosDiag[numrows] = (aQueenBitPosDiag[n] | lsb) << 1LL;
                *pnStack++ = bitfield;
                /* We can't consider positions for the queen which are in the same
                   column, same positive diagonal, or same negative diagonal as another
                   queen already on the board. */
                bitfield = mask & ~(aQueenBitCol[numrows] | aQueenBitNegDiag[numrows] | aQueenBitPosDiag[numrows]);
                ++tree_size;
                continue;
            }
            else
            {

                printf("\n");
                 for(int i = 0; i<board_size;++i){
                      printf(" %d - ", (unsigned int)(log2(aQueenBitRes[i])+1));
                }
                // printf("\n");

                ++g_numsolutions;
                bitfield = *--pnStack;
                --numrows;
                //exit(1);
                continue;
            }
        }
   // }//for odd

    return tree_size;
}



void call_parallel_search(long long board_size, long long cutoff_depth){

    g_numsolutions = 0 ;
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
    }

     if (g_numsolutions != 0)
    {
        printf("PARALLEL SEARCH: size %lld, Tree: %llu,  solutions: %llu\n", board_size, tree_size, g_numsolutions*2);
    }
    else
    {
        printf("No solutions found.\n");
    }
    printf("\n#######################################\n");

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
            /* fill in rightmost 1's in bitfield for half of board_size
               If board_size is 7, half of that is 3 (we're discarding the remainder)
               and bitfield will be set to 111 in binary. */
            bitfield = (1LL << half) - 1LL;
            pnStack = aStack + 1LL; /* stack pointer */

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
           // displayBitsLLU(bitfield);

            if (0ULL == bitfield)
            {
                bitfield = *--pnStack; /* get prev. bitfield from stack */
                
                if (pnStack == aStack) { /* if sentinel hit.... */
                    
                    printf("\nBreak################################################");
                   
                    break ;
                }

                printf("\nBacktracking 0ull- numrows - %lld", numrows-1);
                --numrows;
                continue;
            }

            bitfield &= ~lsb; /* toggle off this bit so we don't try it again */

            printf("\n%llu and bitfield: \n", lsb);
            displayBitsLLU(bitfield);
            printf("numrows: %lld\n", numrows);


            aQueenBitRes[numrows] = lsb; /* save the result */

            if (numrows < board_minus) /* we still have more rows to process? */
            {
                long long int n = numrows++;
                aQueenBitCol[numrows] = aQueenBitCol[n] | lsb;
                aQueenBitNegDiag[numrows] = (aQueenBitNegDiag[n] | lsb) >> 1LL;
                aQueenBitPosDiag[numrows] = (aQueenBitPosDiag[n] | lsb) << 1LL;
                *pnStack++ = bitfield;

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

                ++g_numsolutions;
                bitfield = *--pnStack;

                printf("\nBacktracking sol: \n");
                --numrows;
                //exit(1);
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






/* main routine for N Queens program.*/
int main(int argc, char** argv)
{
    time_t t1, t2;
    int boardsize;

    if (argc != 3) {
        printf("N Queens program by Jeff Somers.\n");
        printf("\tallagash98@yahoo.com or jsomers@alumni.williams.edu\n");
        printf("This program calculates the total number of solutions to the N Queens problem.\n");
        printf("Usage: nq <width of board>\n"); /* user must pass in size of board */
        return 0;
    }

    boardsize = atoi(argv[1]);

    /* check size of board is within correct range */
    if (MIN_BOARDSIZE > boardsize || MAX_BOARDSIZE < boardsize)
    {
        printf("Width of board must be between %d and %d, inclusive.\n",
                                            MIN_BOARDSIZE, MAX_BOARDSIZE );
        return 0;
    }

    time(&t1);
    printf("N Queens program by Jeff Somers.\n");
    printf("\tallagash98@yahoo.com or jsomers@alumni.williams.edu\n");
    printf("Start: \t %s", ctime(&t1));

    
    
    //partial_search(boardsize, (long long)(atoi(argv[2])));

   //call_parallel_search(boardsize, (long long)(atoi(argv[2])));

    // time(&t2);

    // printResults(&t1, &t2);
    g_numsolutions = 0ULL;
   // Nqueen(boardsize);
    completeEnumeration(boardsize);
     if (g_numsolutions != 0)
     {
         printf("For board size %d, %llu solution %s found.\n", boardsize, g_numsolutions, (g_numsolutions == 1 ? "" : "s"));
     }
     else
    {
         printf("No solutions found.\n");
    }

    return 0;
}
