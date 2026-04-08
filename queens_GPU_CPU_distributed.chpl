use CTypes;

use queens_call_serial_search;
use queens_call_mcore_search;
use bitset_serial;
use bitset_mcore_search;
use queens_GPU_single_locale;

//Variables from the command line
config const initial_depth: c_int = 5;
config const second_depth:  c_int = 0;
config const size: uint(16) = 15; //queens
config const prepro: bool = false; //queens first solution

config const scheduler: string = "dynamic";
config const slchunk: int = 1; //chunk used by the final search called by the intermediate search -- for the second level of parallelism.
config const num_threads: int = here.maxTaskPar; //number of threads.
config const mode: string = "serial";
config const data_structure: string = "vector";

config const CPUP: real = 0.0; //CPU percent
config const num_gpus: c_int = 1;
config const language: string = "chpl"; //implementation of the GPU queens search

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
 					queens_call_serial_search(size, mode, prepro);
	 			}
			
	 			when "mcore"{
	 				writeln("--- N-Queens mcore search --- \n\n");
					queens_call_mcore_search(size, initial_depth,scheduler,slchunk,num_threads);
	 			}
                //@todo: the CPU+GPU is not available in this version. 
                when "mgpu"{

	 				writeln("--- N-Queens multi-GPU search - single locale --- \n\n");
	 				GPU_queens_call_search(num_gpus, size,initial_depth,CPUP,slchunk,language);
	 			}
				
				when "multilocale"{

					writeln("--- N-Queens multi-GPU search - multi locale --- ", mode ," -- ", mlsearch,"\n\n");
		 			queens_call_multilocale_search(size,initial_depth,second_depth,scheduler,mode,mlsearch,
		 					lchunk,mlchunk,slchunk,coordinated,pgas,num_threads,profiler,verbose,
		 					CPUP, num_gpus,language);
		 		}//nested
	 		
	
		 		otherwise{
		 			halt("###### ERROR ######\n###### ERROR ######\n###### ERROR ######\n###### WRONG PARAMETERS ######");
		 		}

		 	}//mode
		}//queens vector

	}//lower bound


}
