use CTypes;

use queens_call_serial_search;
use queens_call_mcore_search;
use queens_call_multilocale_search;
use bitset_serial;
use bitset_mcore_search;
use bitset_mlocale_search;

//Variables from the command line
config const initial_depth: c_int = 5;
config const second_depth:  c_int = 7;
config const size: uint(16) = 15; //queens

config const scheduler: string = "dynamic";

//chpl -s queens_mlocale_parameters_parser.GPU=false -s queens_checkPointer=true -s avoidMirrored=true  -s timeDistributedIters=true -s infoDistributedIters=true -M ./modules --fast --no-bounds-checks queens_CPU_distributed.chpl -o  ./bin/queens_distributed.out

config const num_threads: int = here.maxTaskPar; //number of threads.
config const data_structure: string = "vector";
config const mode: string = "mcore";
config const mlsearch: string = "naive";

/////distributed
config const mlchunk:int = 1;
config const slchunk: int = 1; //chunk used by the final search called by the intermediate search -- for the second level of parallelism.
config const lchunk: int = 1; //– The chunk size to yield to each task -- when the iterator uses also the second level of parallelism.
config const coordinated: bool = false; //master process?

config const pgas: bool = false;


config const verbose: bool = false; //verbose network communication
config const profiler: bool = false; //to gather profiler metrics and execution graphics.



proc main(){

	select data_structure {

		when "bitset" {

		 	writeln("\n--- N-QUEENS - Bitset-based data structure --- ");

		 	select mode{

	 			when "serial"{
	 				
 					writeln("--- N-Queens serial search --- \n\n");
					bitset_call_searial_search(size:int);
 					
	 			}
			
	 			when "mcore"{
        			writeln("--- N-Queens mcore search --- \n\n");
        			bitset_call_mcore_search(size,initial_depth,slchunk,num_threads);
    			}
                
                when "mlocale"{
                    writeln(" ########## MLOCALE ############");
                    bitset_call_mlocale_search(size, initial_depth, slchunk, mlchunk, coordinated, num_threads, pgas);
                }
	 		
		 		otherwise{
		 			halt("###### ERROR ######\n###### ERROR ######\n###### ERROR ######\n###### WRONG PARAMETERS ######");
		 		}

		 	}//mode
		}//queens bitset



		when "vector" {

		 	writeln("\n--- N-QUEENS - Vector-based data structure --- ");

		 	select mode{

	 			when "serial"{
	 				
 					writeln("--- N-Queens serial search --- \n\n");
 					queens_call_serial_search(size, mode);
	 			}
			
	 			when "mcore"{
	 				writeln("--- N-Queens mcore search --- \n\n");
					queens_call_mcore_search(size, initial_depth,scheduler,slchunk,num_threads);
	 			}

				when "mlocale"{
					writeln("--- N-Queens  --- ", mode ," -- ", mlsearch,"\n\n");
		 			
					queens_call_multilocale_search(size,initial_depth,second_depth,scheduler,mode,mlsearch,
		 					lchunk,mlchunk,slchunk,coordinated,pgas,num_threads,profiler,verbose,
		 					1.0, 0,"chpl");
				}

		 		otherwise{
		 			halt("###### ERROR ######\n###### ERROR ######\n###### ERROR ######\n###### WRONG PARAMETERS ######");
		 		}

		 	}//mode
		}//queens vector

	}//lower bound


}
