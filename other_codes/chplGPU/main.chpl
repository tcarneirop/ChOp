
use CTypes;


use queens_aux;
use queens_call_mcore_search;
use queens_GPU_single_locale;

//Variables from the command line
config const initial_depth: c_int = 6;

config const size: uint(16) = 12; //queens
config const prepro: bool = false; //queens first solution
//the default coordinated is TRUE
config const scheduler: string = "dynamic";


config const lchunk: int = 1; //chunk for the multicore search



config const num_threads: int = here.maxTaskPar; //number of threads.

config const mode: string = "mgpu";
config const num_gpus: c_int = 0; //if it is not zero, get the number of devices of the system



proc main(){

		 	writeln("\n--- N-QUEENS --- ");
		 	select mode{
		 		when "serial"{
		 			writeln("--- N-Queens serial search --- \n\n");
		 			queens_serial_caller(size, mode, prepro);
		 		}
		 		when "mcore"{
		 			writeln("--- N-Queens mcore search --- \n\n");
		 			queens_node_call_search(size, initial_depth,scheduler,lchunk,num_threads);
		 		}
		 		when "mgpu"{
		 				writeln("--- N-Queens multi-GPU search - single locale --- \n\n");
		 				GPU_queens_call_search(size,initial_depth);
		 		}
		 	}//mode


}
