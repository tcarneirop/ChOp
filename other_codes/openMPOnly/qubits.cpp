#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include <climits>
#include <math.h>
#define _EMPTY_      -1
#define MAX_SIZE 32

double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

typedef struct queen_root{
    unsigned int control;
    int8_t board[MAX_SIZE]; //maximum depth of the solution space.
} QueenRoot;



unsigned long long queens_subproblem_generation(const int m, const int d){

    unsigned int flag = 0;
    int bit_test = 0;
    char board[MAX_SIZE]; 
    int i, depth; 
    unsigned long long int local_tree = 0ULL;
    unsigned long long int num_sol = 0;

    for (i = 0; i < MAX_SIZE; ++i) { //
        board[i] = -1;
    }

    depth = 0;

    do{

        board[depth]++;
        bit_test = 0;
        bit_test |= (1<<board[depth]);


        if(board[depth] == m){
            board[depth] = _EMPTY_;
        }else if ( !(flag &  bit_test ) ){ //it is a valid subsol 
   

                flag |= (1ULL<<board[depth]);
                ++depth;
                ++local_tree;
                if (depth == d){ //handle solution
                   num_sol++;
                   //printf("\nSolution of number: %llu\n\t", num_sol);
                   //for(int i = 0; i<d;++i){
                   //     printf("%d - ", board[i]);
                   //}
                   //printf("\n");

            }else continue;
        }else continue;

        depth--;
        flag &= ~(1ULL<<board[depth]);

    }while(depth >= 0);

    printf("\nNumber of solutions: %llu\n", num_sol);
    return num_sol;
}


int main(int argc, char *argv[]){


    int m;
    int d;

    
    if (argc != 3) {
        printf("Usage: %s <m> <d>\n", argv[0]);
        return 1;
    }


    m = atoi(argv[1]);
    d = atoi(argv[2]);

    if(m<d){
        printf("Usage: m needs to be >= d\n");
        return 1;    
    }

    queens_subproblem_generation(m, d);

    return 0;
}
