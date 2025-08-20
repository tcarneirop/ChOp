
use CTypes;

use fsp_simple_serial;
use fsp_simple_call_mcore_search;
use fsp_simple_call_multilocale_search;

use fsp_johnson_serial;
use fsp_johnson_call_mcore_search;
use fsp_johnson_call_multilocale_search;

//Variables from the command line
config const initial_depth: c_int = 5;
config const second_depth:  c_int = 0;

//the default coordinated is TRUE
config const scheduler: string = "dynamic";
config const lchunk: int = 1; //– The chunk size to yield to each task -- when the iterator uses also the second level of parallelism.
config const slchunk: int = 1; //chunk used by the final search called by the intermediate search -- for the second level of parallelism.
config const mlchunk: int = 1; //Size of the chunk given to each locale. 0 -- uses a heuristic

config const coordinated: bool = false;  //master?
config const verbose: bool = false; //verbose network communication
config const profiler: bool = false; //to gather profiler metrics and execution graphics.
config const pgas: bool = false; //pgas-based active set
config const atype: string = "none"; //atomic type. 'none' when initializing using the optimal -- use like that.

config const num_threads: int = here.maxTaskPar; //number of threads.

config const upper_bound: c_int = 0; //value for the initial upper bound. If it is zero, the optimal solution is going to be used.
config const lower_bound: string = "simple"; //fsp - johnson, fsp - simple, queens, minla

config const mode: string = "mcore";

config const instance: int(8) = 13; //fsp instance

proc main(){

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
				when "single"{
		 		 		writeln("--- CHPL-SIMPLE single depth multi-locale search --- \n");
		 		 		fsp_simple_call_multilocale_search(initial_depth,second_depth,upper_bound,scheduler,
		 		 			lchunk,mlchunk,slchunk,coordinated,pgas,num_threads,profiler,atype,instance,mode,verbose);
		 		}
				when "nested"{
		 		 		writeln("--- CHPL-SIMPLE nested multi-locale search --- \n");
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
                when "single"{
		 		 	writeln("--- CHPL-SIMPLE single depth multi-locale search --- \n");
		 		 	fsp_johnson_call_multilocale_search(initial_depth,second_depth,upper_bound,scheduler,
		 		 		lchunk,mlchunk,slchunk,coordinated,pgas,num_threads,profiler,atype,instance,mode,verbose);
		 		}
				when "nested"{
		 		 	writeln("--- CHPL-Johnson nested multi-locale search --- \n");
		 		 	fsp_johnson_call_multilocale_search(initial_depth,second_depth,upper_bound,scheduler,
		 		 		lchunk,mlchunk,slchunk,coordinated,pgas,num_threads,profiler,atype,instance,mode,verbose);
		 		 }
				otherwise{
					halt("###### ERROR ######\n###### ERROR ######\n###### ERROR ######\n###### WRONG PARAMETERS ######");
				}
			}//mode
		}//johnson bound	
    }
}