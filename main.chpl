
use SysCTypes;

use queens_aux;
use queens_call_mcore_search;
use queens_call_multilocale_search;

use queens_GPU_single_locale;
use GPU_mlocale_utils;

//Variables from the command line

config const initial_depth: c_int = 4;
config const second_depth:  c_int = 7;
config const size: uint(16) = 12; //queens

//the default coordinated is TRUE
config const scheduler: string = "dynamic";

config const mlchunk: int = 0; //inter-node chunk.
config const lchunk: int = 1; //task chunk the inter-node scheduler gives.
config const slchunk: int = 1; //chunk for the second level of parallelism. 

config const coordinated: bool = true;  //centralized node?
config const pgas: bool = false; //pgas-based active set?
config const num_threads: int = here.maxTaskPar; //number of threads. 

config const profiler: bool = false; //to gather profiler metrics and chplvis graphs. 

config const atype: string = "none"; //atomic type. 'none' when initializing using the optimal -- use like that.

config const verbose: bool = false; //verbose network communication 

config const real_number_computers: int = 1;
config const mode: string = "improved";
config const mlsearch: string = "mlocale";
config const num_gpus: c_int = 0; //Get the number of devices of the system if it is not zero

config const CPUP: real = 0.0; //CPU percent


proc main(){


		writeln("--- N-QUEENS --- ");

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

				queens_call_multilocale_search(size,initial_depth,second_depth,scheduler,mode,mlsearch,
							lchunk,mlchunk,slchunk,coordinated,pgas,num_threads,profiler,verbose, CPUP, num_gpus);
			}//improved
			when "mgpu"{
				writeln("--- N-Queens multi-GPU search - single locale --- \n\n");
				GPU_queens_call_search(size,initial_depth,CPUP,lchunk);
			}
			otherwise{
				halt("###### ERROR ######\n###### ERROR ######\n###### ERROR ######\n###### WRONG PARAMETERS ######");
			}

		}//mode

}