
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <cmath>
#include <omp.h>
#include <sys/time.h>
#include <math.h>
#include <tgmath.h>

#ifndef MAX_BOARDSIZE
    #define  MAX_BOARDSIZE 32
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

inline unsigned long long factorial(unsigned long long n){
    return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
}

void check(unsigned long long m, unsigned long long d, unsigned long long nsols){
    unsigned long long mfat = factorial(m);
    unsigned long long mmdfat = factorial(m-d);

    if(mfat/(mmdfat)!=nsols){
        printf("\n############ ERROR - WRONG NUMBER OF SOLS #############\n");
        exit(1);
    }else{
        printf("\n############ NUM SOLS OK #############\n");
        
    }

}


unsigned long long partial_search_64(const long long m, const long long d)
{

    long long aQueenBitRes[MAX_BOARDSIZE];
    long long aQueenBitCol[MAX_BOARDSIZE]; 
    long long aStack[MAX_BOARDSIZE]; 
     
    long long int *pnStack;

    long long int pnStackPos = 0LLU;

    long long numrows = 0LL; 
    unsigned long long lsb; 
    unsigned long long bitfield; 
    long long i;
    
    long long mask = (1LL << m) - 1LL; 
    
    unsigned long long tree_size = 0ULL;
    /* Initialize stack */
    aStack[0] = -1LL; /* set sentinel -- signifies end of stack */


        bitfield = 0ULL;

        bitfield = (1LL << m) - 1LL;
        pnStack = aStack + 1LL; 
        
        pnStackPos++;

        aQueenBitRes[0] = 0LL;
        aQueenBitCol[0] = 0LL;

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
            aQueenBitRes[numrows] = 63 - __builtin_clzll(lsb); 
            
            if (numrows < d) {
                long long n = numrows++;
                aQueenBitCol[numrows] = aQueenBitCol[n] | lsb;
                
                pnStackPos++;
                
                *pnStack++ = bitfield;
                
                bitfield = mask & ~(aQueenBitCol[numrows]);
                
                ++tree_size;

                if(numrows == d){
                 
                    ++g_numsolutions;

                    #ifdef PRINT
                        printf("\nSolution of number %llu - \n\t",g_numsolutions);
                        
                        printf("[ ");
                        for(long long s = 0; s<d;++s){
                            printf("%llu ", aQueenBitRes[s]);
                            if(s<d-1)
                                printf(" - ");
                        }
                        printf("]");
                    #endif

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
   
    printf("\n \nNum sols: %llu \n", g_numsolutions);

#ifdef PRINT
    check( m, d, g_numsolutions);
#endif

    return tree_size;

}


/* main routine for N Queens program.*/
int main(int argc, char** argv)
{

    int m = atoi(argv[1]);
    int d = atoi(argv[2]);
    if(d>m){
        printf("############## ERROR: m needs to be >= d ###############");
        exit(1);
    }
        //exec, size, search, depth, chunk; 
        partial_search_64((unsigned long long)m,(unsigned long long)d);
    return 0;
}



