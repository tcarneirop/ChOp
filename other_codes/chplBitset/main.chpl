use CTypes;
use Time;
use bitset_serial;
use bitset_mcore_search;
use bitset_subproblem_module;
use bitset_subproblem_explorer;
use bitset_partial_search;
use bitset_mlocale_search;


config const size: int = 12;
config const initial_depth: int = 2;
config const second_depth:  c_int = 7;

config const slchunk:int = 8;
config const mlchunk:int = 1;
config const num_threads: int = here.maxTaskPar; //number of threads.
config const coordinated: bool = false;
config const pgas: bool = false;
config const mode: string = "mcore";

select mode{
    when "mlocale"{
        writeln(" ########## MLOCALE ############");

        bitset_call_mlocale_search(size, initial_depth, slchunk, mlchunk, coordinated, num_threads, pgas);

        
    }
    when "mcore"{
        writeln(" ########## MCORE ############");
        bitset_call_mcore_search(size,initial_depth,slchunk,num_threads);
    }
}
