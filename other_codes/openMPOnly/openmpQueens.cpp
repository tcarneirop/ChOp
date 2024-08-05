#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include <climits>

#define _EMPTY_      -1


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
    int8_t board[12]; //maximum depth of the solution space.
} QueenRoot;



inline bool queens_is_legal_placement(const char *__restrict__ board, const int r)
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



inline bool queens_is_legal_placement_depth(const char *__restrict__ board, const int r, const int initial_depth)
{

    int i;
    int ld;
    int rd;
    // Check vertical
    for ( i = initial_depth-1; i < r; ++i)
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

inline void queens_keep_subproblem(QueenRoot *__restrict__ root_prefixes,unsigned int flag,
    char *__restrict__  board,int initialDepth,int num_sol){

    root_prefixes[num_sol].control = flag;

    for(int i = 0; i<initialDepth;++i)
      root_prefixes[num_sol].board[i] = (char)board[i];
}


unsigned long long queens_subproblem_generation(int size, int initialDepth,
    unsigned long long *__restrict__ tree_size, QueenRoot *__restrict__ root_prefixes){

    unsigned int flag = 0;
    int bit_test = 0;
    char board[32]; 
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
            board[depth] = _EMPTY_;
        }else if ( !(flag &  bit_test ) && queens_is_legal_placement(board, depth) ){ //it is a valid subsol 
   
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
                   queens_keep_subproblem(root_prefixes,flag,board,initialDepth,num_sol);
                   num_sol++;
            }else continue;
        }else continue;

        depth--;
        flag &= ~(1ULL<<board[depth]);

    }while(depth >= 0);

    *tree_size = local_tree;

    return num_sol;
}

void queens_subtree_enumeration(const unsigned idx, const int N, const unsigned nPrefixes, 
    const int initial_depth, QueenRoot *__restrict__ root_prefixes,
    unsigned long long int *__restrict__ vector_of_tree_size, 
    unsigned long long int *__restrict__ sols){


    unsigned int flag = 0;
    char board[24];
    int N_l = N;
    int i, depth;
    unsigned long long  qtd_sols_thread = 0ULL;
    int depthGlobal = initial_depth;
    unsigned long long int tree_size = 0ULL;

    for (i = 0; i < N_l; ++i) {
        board[i] = _EMPTY_;
    }

    flag = root_prefixes[idx].control;

    for (i = 0; i < depthGlobal; ++i)
        board[i] = root_prefixes[idx].board[i];

    depth=depthGlobal;

    do{

        board[depth]++;
        const int mask = 1<<board[depth];

        if(board[depth] == N_l){
            board[depth] = _EMPTY_;
            depth--;
            flag &= ~(1<<board[depth]);
        }else if (!(flag &  mask ) && queens_is_legal_placement_depth(board, depth,depthGlobal)){

                ++tree_size;
                flag |= mask;

                depth++;

                if (depth == N_l) { //sol
                    ++qtd_sols_thread ;

                    depth--;
                    flag &= ~mask;
                }
            }
        }while(depth >= depthGlobal); 

    sols[idx] = qtd_sols_thread ;
    vector_of_tree_size[idx] = tree_size;

}//kernel
////////


void call_queens(int size, int initialDepth, int chunk){


    unsigned long long initial_tree_size = 0ULL;
    unsigned long long qtd_sols_global = 0ULL;
    unsigned long long total_tree_size = 0ULL;

    unsigned nMaxPrefixes = 95580635;
    int num_gpus = 0;


    unsigned long long thread_load[omp_get_max_threads()];
    for(int i = 0; i<omp_get_max_threads();++i)
        thread_load[i] = 0ULL;
  
    QueenRoot* subproblems_pool = (QueenRoot*)malloc(sizeof(QueenRoot)*nMaxPrefixes);
    unsigned long long int *vector_of_tree_size_h = (unsigned long long int*)malloc(sizeof(unsigned long long int)*nMaxPrefixes);
    unsigned long long int *solutions_h = (unsigned long long int*)malloc(sizeof(unsigned long long int)*nMaxPrefixes);

    double initial_time = rtclock();

    //initial search, getting Feasible, Valid and Incomplete solutions -- subproblems;
    unsigned long long n_subproblems = queens_subproblem_generation((short)size, initialDepth, &initial_tree_size, subproblems_pool);


    printf("\n### Queens size: %d, Initial depth: %d - Num_explorers: %llu - num_threads: %d", size, initialDepth,n_subproblems, omp_get_max_threads());

    #pragma omp parallel for schedule(runtime) default(none) shared(chunk,size, thread_load,n_subproblems, initialDepth, subproblems_pool, vector_of_tree_size_h, solutions_h)
    for(unsigned long long subproblem = 0; subproblem<n_subproblems; ++subproblem){
        int id = omp_get_thread_num();
        queens_subtree_enumeration(subproblem, size, n_subproblems, initialDepth, subproblems_pool,vector_of_tree_size_h, solutions_h);
        #ifdef REPORT
        thread_load[id] += vector_of_tree_size_h[subproblem];
        #endif
    } 
 
    //Reducing the metrics
    for(int i = 0; i<n_subproblems;++i){
        qtd_sols_global += solutions_h[i];
        total_tree_size +=vector_of_tree_size_h[i];
    }
    double final_time = rtclock();

    #ifdef REPORT
    printf("\nThread load report: \n");
    unsigned long long biggest = 0;
    unsigned long long smallest = ULLONG_MAX;
    for(int id = 0; id<omp_get_max_threads();++id){
        if(thread_load[id]<smallest)
            smallest = thread_load[id];
        if(thread_load[id]>biggest)
            biggest = thread_load[id];
        printf("\t Thread Id: %d - Load: %llu - Percent: %.2f\n", id,thread_load[id],((double)thread_load[id]/(double)total_tree_size)*100.0);
    }
    #endif

    #ifdef IMPROVED
       qtd_sols_global*=2;
    #endif


    printf("\n\nInitial tree size: %llu", initial_tree_size );
    printf("\nParallel Tree size: %llu\nTotal tree size: %llu\nNumber of solutions found: %llu.\n", total_tree_size,(initial_tree_size+total_tree_size),qtd_sols_global );
    printf("\nElapsed total: %.3f\n", (final_time-initial_time));
    #ifdef REPORT
    printf("\n\tBiggest thread load: %llu", biggest);
    printf("\n\tSmallest thread load: %llu", smallest);
    printf("\n\tBiggest/smallest: %.3fx\n", (double)biggest/(double)smallest);
    #endif

}


int main(int argc, char *argv[]){


    int size;
    int chunk;
    int initialDepth;

    #ifdef IMPROVED
        printf("### IMPROVED SEARCH - Avoiding mirrored solutions\n");
    #endif

    if (argc != 4) {
        printf("Usage: %s <size> <initial depth>\n", argv[0]);
        return 1;
    }

    size = atoi(argv[1]);
    initialDepth = atoi(argv[2]);
    chunk = atoi(argv[3]);

    call_queens(size, initialDepth,chunk);

    return 0;
}
