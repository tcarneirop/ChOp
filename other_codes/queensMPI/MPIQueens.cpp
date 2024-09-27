#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include <climits>
#include <cfloat>
#include <mpi.h>
#include <climits>

#define _EMPTY_      -1

/// this is used to check if the solution produced is correct
unsigned long long check_sols_number[] = {0,	0,	0,	2,	10,	4,	40,	92,	352,	724,	2680,	14200,	73712,	
365596,	2279184,	14772512,	95815104,	666090624,	4968057848,	39029188884,314666222712,2691008701644,24233937684440,227514171973736 };

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
    //for ( i = 0; i < r; ++i)
    //    if (board[i] == board[r]) return false;
    // Check diagonals
    ld = board[r];  //left diagonal columns
    rd = board[r];  // right diagonal columns
    for ( i = r-1; i >= 0; --i) {
      --ld; ++rd;
      if (board[i] == ld || board[i] == rd) return false;
    }

    return true;
}


inline void queens_keep_subproblem(QueenRoot *__restrict__ root_prefixes,unsigned int flag,char *__restrict__  board,int initialDepth,int num_sol){

    root_prefixes[num_sol].control = flag;

    for(int i = 0; i<initialDepth;++i)
      root_prefixes[num_sol].board[i] = (char)board[i];
}


unsigned long long int queens_subproblem_generation(const int size, const int initialDepth,
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
                //if(block_ub > upper)   block_ub = upper;
        }else if (  !(flag &  bit_test ) && queens_is_legal_placement(board, depth)  ){ //it is a valid subsol 
   
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
        }else if (!(flag &  mask ) && queens_is_legal_placement(board, depth)){

                ++tree_size;
                flag |= mask;

                depth++;

                if (depth == N_l) { //sol
                    ++qtd_sols_thread ;

                    depth--;
                    flag &= ~mask;
                }
            }
        }while(depth >= depthGlobal); //FIM DO DFS_BNB

    sols[idx] = qtd_sols_thread ;
    vector_of_tree_size[idx] = tree_size;

}//kernel
////////

unsigned long long queens_get_rank_load(int mpi_rank, int num_ranks, unsigned long long num_subproblems){
    unsigned long long rank_load = num_subproblems/num_ranks;
    return (mpi_rank == (num_ranks-1) ? rank_load + (num_subproblems % num_ranks): rank_load);
}


void queens_call_mpi_queens(int size, int initialDepth, int mpi_rank, int num_ranks){


    unsigned long long initial_tree_size = 0ULL;
    unsigned long long rank_num_sols = 0ULL;
    unsigned long long rank_tree_size = 0ULL;


    unsigned long long global_num_sols = 0ULL;
    unsigned long long global_tree_size = 0ULL;
    double             global_exec_time = 0.;

    unsigned int nMaxPrefixes = 95580635;

  
    QueenRoot* root_prefixes_h = (QueenRoot*)malloc(sizeof(QueenRoot)*nMaxPrefixes);
    //initial search, getting Feasible, Valid and Incomplete solutions -- subproblems;
    unsigned long long n_subproblems = queens_subproblem_generation(size, initialDepth ,&initial_tree_size, root_prefixes_h);
    
    unsigned long long rank_load = queens_get_rank_load(mpi_rank,num_ranks, n_subproblems);
    root_prefixes_h = root_prefixes_h + n_subproblems/num_ranks*mpi_rank;

    unsigned long long int *vector_of_tree_size_h = (unsigned long long int*)malloc(sizeof(unsigned long long int)*rank_load);
    unsigned long long int *solutions_h = (unsigned long long int*)malloc(sizeof(unsigned long long int)*rank_load);

    double rank_initial_time = rtclock();


    if(mpi_rank == 0){
        #ifdef IMPROVED
        printf("### IMPROVED SEARCH - Avoiding mirrored solutions\n");
        #endif
        printf("\n### Queens size: %d, Initial depth: %d - rank_load: %llu - num_threads: %d", size, initialDepth,rank_load, omp_get_max_threads());
    }
    //printf("\n### Queens size: %d, Initial depth: %d - rank_load: %llu - num_threads: %d", size, initialDepth,rank_load, omp_get_max_threads());
   
    #pragma omp parallel for schedule(runtime) default(none) shared(size,rank_load, initialDepth, root_prefixes_h, vector_of_tree_size_h, solutions_h)
    for(unsigned long long subproblem = 0; subproblem<rank_load; ++subproblem){
        queens_subtree_enumeration(subproblem, size, rank_load, initialDepth, root_prefixes_h,vector_of_tree_size_h, solutions_h);
    } 

    //Reducing the metrics
    for(int i = 0; i<rank_load;++i){
        rank_num_sols += solutions_h[i];
        rank_tree_size += vector_of_tree_size_h[i];
    }
   
    //printf("\n Rank %d Tree size: %llu", mpi_rank,rank_tree_size);

    MPI_Allreduce(&rank_tree_size, &global_tree_size, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&rank_num_sols, &global_num_sols, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        
    double rank_final_time = rtclock();
    double rank_total_rime = rank_final_time - rank_initial_time;

    MPI_Allreduce(&rank_total_rime, &global_exec_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);


    if(mpi_rank == 0){
        #ifdef IMPROVED
        global_num_sols*=2;
        #endif
        printf("\nFinal Tree size: %llu\nNumber of solutions found: %llu\n", global_tree_size+initial_tree_size,global_num_sols);
        printf("\nElapsed total: %.3f\n", global_exec_time);
    }


    #ifdef CHECKSOL
        if(mpi_rank == 0){
            if(global_num_sols == check_sols_number[size-1])
                printf("\n####### SUCCESS - CORRECT NUMBER OF SOLS. FOR SIZE %d\n", size);
            else
                printf("########## ERROR -- INCORRECT NUMBER FOS SOLS. FOR SIZE %d\n", size);
        }
    #endif



    #ifdef RANKLOADS

    unsigned long long rank_loads[num_ranks];
    double rank_exec_times[num_ranks];
    
    MPI_Gather(&rank_tree_size, 1, MPI_UNSIGNED_LONG_LONG, rank_loads, 1, MPI_UNSIGNED_LONG_LONG, 0,MPI_COMM_WORLD);
    MPI_Gather(&rank_total_rime, 1, MPI_DOUBLE, rank_exec_times, 1, MPI_DOUBLE, 0,MPI_COMM_WORLD);
    
    if( mpi_rank == 0 ){
        unsigned long long biggest = 0;
        
        int biggest_load;
        int smallest_load;

        unsigned long long smallest = ULLONG_MAX;
      
        
        printf("\n\n########## Per-rank Load Report: <rank> <tree_size> <exec_time> \n");
        for(int rank = 0; rank<num_ranks;++rank){
            
            if (rank_loads[rank] < smallest) {smallest = rank_loads[rank]; smallest_load = rank;}
            if (rank_loads[rank] > biggest)  {biggest = rank_loads[rank]; biggest_load = rank;} 


            printf("\tRank %d - load: %llu - %.3f\n", rank, rank_loads[rank], rank_exec_times[rank] );

        }

        printf("\n\tBiggest load: %llu -  %.3f ", rank_loads[biggest_load], rank_exec_times[biggest_load] );
        printf("\n\tSmallest load: %llu - %.3f ", rank_loads[smallest_load], rank_exec_times[smallest_load] );
        printf("\n\tBiggest/Smallest: %.3f\n", (double)rank_loads[biggest_load]/(double)rank_loads[smallest_load]);

    }
    #endif


}


int main(int argc, char *argv[]){


    int size;
    int initialDepth;
    int mpi_rank;
    int num_ranks;

   
    if (argc != 4) {
        printf("Usage: %s <size> <initial depth> <chunk>\n", argv[0]);
        return 1;
    }

    size = atoi(argv[1]);
    initialDepth = atoi(argv[2]);
    
    MPI_Init(&argc, &argv);
    
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    
    if(mpi_rank == 0) printf("\nNumber of MPI Ranks: %d", num_ranks);
    
    queens_call_mpi_queens(size, initialDepth,mpi_rank,num_ranks);


    MPI_Finalize();

    return 0;
}
