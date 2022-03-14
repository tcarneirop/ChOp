#ifndef CALL_QUEENS_MPI_H
#define CALL_QUEENS_MPI_H

#include "../queen_prefixes.hh"
#include "../queen_mcore.hh"
#include "../classTimer.hh"
#include <omp.h>

#define __STATIC__  0
#define __DYNAMIC__ 1
 

unsigned long long  queens_calculaNPrefixos(int nVertice, int nivelPrefixo) {

    unsigned long long  x = 1ULL;
    int i;


    if(nivelPrefixo==1)
        return (unsigned long long)nVertice;

    for (i = 0; i < nivelPrefixo; ++i) {
        x *= ((unsigned long long)(nVertice - i));
    }
    return x;
}



unsigned long long get_mpi_chunk(int proc_id, unsigned long long survivors, int num_procs){
	
	unsigned long long id = (unsigned long long)proc_id;
	unsigned long long chunk = survivors/num_procs;
	

	if(proc_id == num_procs-1)
		chunk+=(survivors%num_procs);

	return chunk;
}



unsigned long long range_start(int proc_id, unsigned long long survivors, int num_procs){

	unsigned long long id = (unsigned long long)proc_id;

	return (id * (survivors/num_procs) );
}


unsigned long long range_end(int proc_id, unsigned long long survivors, int num_procs){
	
	unsigned long long id = (unsigned long long)proc_id;
	unsigned long long chunk = get_mpi_chunk(proc_id,survivors,num_procs);
	unsigned long long start = range_start(proc_id,survivors, num_procs);

	return start + chunk;
}



void call_queens_mpi(int proc_id, int num_procs, int size, int initialDepth, int load_b){

	double initial_time = rtclock();
	double final_time;
	double elapsed_total;
	
	unsigned long long int tree_size = 0ULL;
	unsigned long long int initial_tree_size = 0ULL;
	unsigned long long int global_tree = 0ULL;
	// unsigned long long int thread_reduc_tree = 0ULL;

	double report[3];
   
	unsigned int nMaxPrefixos = queens_calculaNPrefixos(size,initialDepth);

	QueenRoot *root_prefixes = (QueenRoot*)malloc(sizeof(QueenRoot)*nMaxPrefixos);

	int n_threads = omp_get_num_procs();
	int *thread_tree = (int*) calloc(n_threads,sizeof(int));

	printf("\nNum threads: %d.\n",n_threads);

	unsigned int  qtd_sols_global = 0;



	double survivors_initial_time = rtclock();

	unsigned int survivors = BP_queens_prefixes(size,initialDepth ,&tree_size,root_prefixes);

	double survivors_final_time = rtclock();


	double survivor_time = survivors_final_time-survivors_initial_time;

	int *sols = (int*)malloc(sizeof(int)*survivors); 
  	int *vector_of_tree_size = (int*)malloc(sizeof( int)*survivors);

  	//mpi variables
  	//mpi variables
  	//mpi variables
  	unsigned long long r_start = range_start(proc_id,survivors,num_procs);
  	unsigned long long r_end = range_end(proc_id,survivors, num_procs);
  	unsigned long long chunk = get_mpi_chunk(proc_id,survivors,num_procs);



	printf("\n Number of threads now: %d", omp_get_num_threads());
	printf("\n Number of processors: %d", omp_get_num_procs());
	printf("\n Maximum number of threads: %d", omp_get_max_threads());

	printf("\n\nSurvivors: %llu.\n\t Local chunk for process %d: %llu.\n\tRange start: %llu - Range end: %llu.\n\n", 
		survivors,proc_id,chunk,r_start,r_end);
 	



	if(load_b == __STATIC__){
			printf("\n\nStatic: \n\n");
			#pragma omp parallel for num_threads( omp_get_num_procs() ) schedule(static)
			for(int idx = r_start; idx<r_end ;++idx){
				int tid=omp_get_thread_num();
				BP_queens_MC_root_dfs(size, idx, chunk,initialDepth,root_prefixes,vector_of_tree_size, sols);
				//queens_mcore_BP_dfs(size,idx,vector_of_flags,path,survivors,initialDepth,sols,vector_of_tree_size);
				thread_tree[tid]+=vector_of_tree_size[idx];
			}
	}
	else{

		if(load_b == __DYNAMIC__){
				printf("\n\nDynamic: \n\n");
				#pragma omp parallel for num_threads( omp_get_num_procs() ) schedule(dynamic)
				for(int idx = r_start; idx<r_end ;++idx){
					int tid=omp_get_thread_num();
					BP_queens_MC_root_dfs(size, idx, chunk,initialDepth,root_prefixes,vector_of_tree_size, sols);
					//queens_mcore_BP_dfs(size,idx,vector_of_flags,path,survivors,initialDepth,sols,vector_of_tree_size);
					thread_tree[tid]+=vector_of_tree_size[idx];
				}

		}//if dynamic
	}//if


	for(int i = 0; i<survivors; ++i)
    	qtd_sols_global+=sols[i];


	initial_tree_size = tree_size;
	tree_size = 0ULL;

	for(int i = 0; i<survivors; ++i){
	 	if(vector_of_tree_size[i]>0){
	 		tree_size+=vector_of_tree_size[i];
	 	}
	}


	final_time = rtclock();
	elapsed_total = final_time - initial_time;


	
	
	
	MPI_Reduce(&tree_size, &global_tree, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0,
           MPI_COMM_WORLD);


	
	MPI_Barrier(MPI_COMM_WORLD);
	if(proc_id == 0){
		printf("\nSize: %d x %d.\n", size,size);
		printf("\tDepth of initial search: %d.\n",initialDepth);
		printf("\tMaximum number of nodes at depth %d: %lu.\n",initialDepth,nMaxPrefixos);
		printf("\tSurvivors: %d, i.e, %.2f %% of the maximum number. \n", survivors,((float)survivors/(float)nMaxPrefixos)*100);
		printf("\n\tSerial tree size: %llu", initial_tree_size );
		printf("\n\tMulticore tree size: %llu", tree_size);
		printf("\n\tTree size: %llu\n\n", initial_tree_size+global_tree);
		printf("\nExecution time: %.3f \n\n",elapsed_total);
	}
	

	
}//end of queens mpi



#endif