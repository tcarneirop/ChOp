use CTypes;

use queens_call_serial_search;
use queens_call_mcore_search;
use queens_GPU_single_locale;
use queens_call_multilocale_search;

use bitset_serial;
use bitset_mcore_search;
use bitset_mlocale_search;



//Variables from the command line
config const initial_depth: c_int = 2;
config const second_depth:  c_int = 5;
config const size: uint(16) = 15; //queens

config const scheduler: string = "dynamic";
config const num_threads: int = here.maxTaskPar; //number of threads.
config const mode: string = "mcore";
config const data_structure: string = "vector";

config const CPUP: real = 0.0; //CPU percent
config const num_gpus: c_int = 1;
config const language: string = "chpl"; //implementation of the GPU queens search

config const verbose: bool = false; //verbose network communication
config const profiler: bool = false; //to gather profiler metrics and execution graphics.

/////distributed
config const mlchunk:int = 1;
config const slchunk: int = 1; //chunk used by the final search called by the intermediate search -- for the second level of parallelism.
config const lchunk: int = 1; //– The chunk size to yield to each task -- when the iterator uses also the second level of parallelism.
config const coordinated: bool = false; //master process?
config const mlsearch: string = "naive";

config const pgas: bool = false;


proc main(){

	forall loc in Locales {
	  on loc {

	    const num_gpus = here.gpus.size;

	    coforall gpu_id in 0..<num_gpus {
	
	      var new_gpu_id: c_int = (here.id:c_int * num_gpus:c_int + gpu_id:c_int) % num_gpus:c_int;
	      
	      on here.gpus[new_gpu_id:int] {
	        var warmup_array: [1..64] int;
	        foreach i in 1..64 {
	          warmup_array[i] = i;
	        }
	      }
	    }
	  }
	}

	
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
                //@todo: the CPU+GPU is not available in this version, it goes to mlocale -- each gpu is a rank and the cpu itself gets a rank
                when "mgpu"{

	 				writeln("--- N-Queens multi-GPU search - single locale --- \n\n");
	 				GPU_queens_call_search(num_gpus, size,initial_depth,CPUP,slchunk,language);
	 			}
				
				when "mlocale"{

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
