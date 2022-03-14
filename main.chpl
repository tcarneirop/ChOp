
use SysCTypes;

use fsp_simple_serial;
use fsp_simple_call_mcore_search;
use fsp_simple_call_inital_search;
use fsp_simple_call_multilocale_search;

use fsp_johnson_serial;
use fsp_johnson_call_mcore_search;
use fsp_johnson_call_initial_search;
use fsp_johnson_call_multilocale_search;


use parametrization_local_search;
use queens_aux;
use queens_call_mcore_search;
use queens_call_multilocale_search;

use queens_GPU_single_locale;
use GPU_mlocale_utils;

use ITE_queens_chpl;


use parameters_record;

//Variables from the command line
config const initial_depth: c_int = 2;
config const second_depth:  c_int = 7;
config const size: uint(16) = 12; //queens

//the default coordinated is TRUE
config const scheduler: string = "dynamic";

config const mlchunk: int = 0; //inter-node chunk.
config const lchunk: int = 1; //task chunk the inter-node scheduler gives.
config const slchunk: int = 1; //chunk for the second level of parallelism. 

config const coordinated: bool = true;  //centralized node?
//available modes:
/// mlocale:
/// improved:
///	sgpu: single-gpu execution
/// cpu-gpu: uses all CPUs and GPUs of the locale at once.

config const pgas: bool = false; //pgas-based active set?

config const num_threads: int = here.maxTaskPar; //number of threads. 
config const profiler: bool = false; //to gather profiler metrics and execution graphics. 
config const number_exec: int = 1;   //going to be removed soon. 
config const upper_bound: c_int = 0; //value for the initial upper bound. If it is zero, the optimal solution is going to be used. 
config const lower_bound: string = "johnson"; //type of lowerbound. Johnson and simple.
config const atype: string = "none"; //atomic type. 'none' when initializing using the optimal -- use like that.
config const instance: int(8) = 13; //fsp instance

config const verbose: bool = false; //verbose network communication 

config const heuristic: string = "none";
config const problem: string = "simple"; //fsp - johnson, fsp - simple, queens, minla
config const computers: int = 1;

config const mode: string = "improved";
config const mlsearch: string = "mlocale";
config const num_gpus: c_int = 0; //if it is not zero, get the number of devices of the system
config param build_gpu_code: bool = true;
config param build_mlocale_code: bool = true;

config const CPUP: real = 0.0; //CPU percent



proc main(){

	//@todo -- these chunks are confusing..
	if(heuristic!="none") then initialization(heuristic,lower_bound, instance, mode);
	else{

	select lower_bound {
		when "simple"{//using simple bound
			select mode{
				 when "serial"{
				 	writeln("--- CHPL-SIMPLE serial search --- \n\n");
				 	fsp_simple_call_serial(upper_bound,instance);
				 }
				when "mcore"{
					writeln(" --- CHPL-SIMPLE mcore search --- \n\n");
					fsp_simple_call_multicore_search(initial_depth,upper_bound,scheduler,lchunk,num_threads,instance);
				}
			
				when "improved"{
				 		writeln("--- CHPL-SIMPLE IMPROVED multi-locale search --- \n");
				 		fsp_simple_call_multilocale_search(initial_depth,second_depth,upper_bound,scheduler,
				 			lchunk,mlchunk,slchunk,coordinated,pgas,num_threads,profiler,atype,instance,mode,verbose);
				}
				
				otherwise{
					halt("###### ERROR ######\n###### ERROR ######\n###### ERROR ######\n###### WRONG PARAMETERS ######");
				}
			} 
		}//end of simple bound
		when "johnson"{
			writeln("\n --- JOHNSON LOWER BOUND --- ");
			select mode{
				when "serial"{
				 	writeln("--- CHPL-Johnson serial search --- \n\n");
				 	fsp_johnson_call_serial(upper_bound, instance);
				 }//serial
				when "mcore"{
					writeln("--- CHPL-Johnson mcore search --- \n\n");
					fsp_johnson_call_multicore_search(initial_depth,upper_bound,scheduler,lchunk,num_threads,instance,true);
				}//mcode
				
				when "improved"{
				 	writeln("--- CHPL-Johnson IMPROVED multi-locale search --- \n");
				 	fsp_johnson_call_multilocale_search(initial_depth,second_depth,upper_bound,scheduler,
				 		lchunk,mlchunk,slchunk,coordinated,pgas,num_threads,profiler,atype,instance,mode,verbose);
				 }//johnson improved
		
				otherwise{
					halt("###### ERROR ######\n###### ERROR ######\n###### ERROR ######\n###### WRONG PARAMETERS ######");
				}
			}//mode
		}//johnson bound
		when "queens"{

		 	writeln("\n--- N-QUEENS --- ");
		 	select mode{
		 		when "serial"{
		 			writeln("--- N-Queens serial search --- \n\n");
		 			queens_parser(size);
		 		}
		 		when "mcore"{
		 			writeln("--- N-Queens mcore search --- \n\n");
		 			queens_node_call_search(size, initial_depth,scheduler,slchunk,num_threads);
		 		}
		 		when "improved"{
		 			writeln("--- N-Queens  --- ", mlsearch ,"\n\n");
		 				queens_call_multilocale_search(size,initial_depth,second_depth,scheduler,mode,mlsearch,
		 					lchunk,mlchunk,slchunk,coordinated,pgas,num_threads,profiler,verbose,
		 					CPUP, num_gpus);
					
		 		}//improved
				
		 			when "mgpu"{
		 				writeln("--- N-Queens multi-GPU search - single locale --- \n\n");
		 				GPU_queens_call_search(size,initial_depth,CPUP,lchunk);
		 			}
				
		 		otherwise{
		 			halt("###### ERROR ######\n###### ERROR ######\n###### ERROR ######\n###### WRONG PARAMETERS ######");
		 		}

		 	}//mode
		}//queens
	
	}//lower bound

}
}

