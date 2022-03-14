#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "call_queens_mpi.h"


int main(int argc, char *argv[]){

	int initialDepth;
    int size;


    MPI_Init(NULL, NULL);
    int num_procs;
    int proc_id;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    int load_b; 

    if(argc<3){
        
        printf("\nWrong Parameters.\n");
        exit(1);
    
    }
    else{


        size = atoi(argv[1]);
        initialDepth = atoi(argv[2]);
        load_b = atoi(argv[3]);

        printf("\n MPI-Queens. \n\tSize: %d\n\tInitial Depth: %d\n", size,initialDepth);
        // Get the rank of the process
        MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
        // Get the number of processes
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);


    
        MPI_Get_processor_name(processor_name, &name_len);
        printf("Hello world from processor %s, rank %d out of %d processors\n",
        processor_name, proc_id, num_procs);
        
        call_queens_mpi(proc_id,num_procs,size,initialDepth,load_b);    
    }




    MPI_Finalize();

    return 0;
}