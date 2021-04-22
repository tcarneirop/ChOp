
use GPUIterator;
use GPU_aux;
use CPtr;
use queens_prefix_generation;
use queens_node_module;
use queens_aux;
use Time;

use BlockDist;
use VisualDebug;
use CommDiagnostics;
use DistributedIters;


use SysCTypes;

config const size: uint(16) = 12;
config const initial_depth: c_int = 7;
config const distributed: bool = false;

///////////////////////////////////////////////////////////////////////////
//C-Interoperability
///////////////////////////////////////////////////////////////////////////

require "headers/GPU_queens.h";


extern proc GPU_call_cuda_queens(size: uint(16), initial_depth:c_int, n_explorers:c_uint, 
		active_set_h: c_ptr(queens_node),vector_of_tree_size_h: c_ptr(c_uint), 
		sols_h: c_ptr(c_uint)): void;


///////////////////////////////////////////////////////////////////////////
//CUDA Vectors
///////////////////////////////////////////////////////////////////////////

var vector_of_tree_size_h: [0..#75580635] c_uint;
var sols_h: [0..#75580635] c_uint;

//var active_set_h: [0..#] queens_node;
var maximum_number_prefixes: int = 75580635;
var active_set_h: [0..#maximum_number_prefixes] queens_node;



var n_explorers : int; //MUST BE INT!
	

///////////////////////////////////////////////////////////////////////////
//Single-locale wrapper
///////////////////////////////////////////////////////////////////////////

var GPUWrapper = lambda (lo:int, hi: int, n_explorers: int) {
  		GPU_call_cuda_queens(size, initial_depth, n_explorers:c_uint, 
		c_ptrTo(active_set_h),c_ptrTo(vector_of_tree_size_h), c_ptrTo(sols_h));
};


///////////////////////////////////////////////////////////////////////////
//Generating the initial Pool of nodes and metrics
///////////////////////////////////////////////////////////////////////////

var initial_tree_size : uint(64) = 0;
var final_tree_size : uint(64) = 0;
var final_sol: uint(64) = 0;
var initial, final, bulktransfer: Timer;


//search metrics
var metrics = (0:uint(64),0:uint(64));

//starting the search
initial.start();

//generating the initial pool of nodes
metrics+= queens_node_generate_initial_prefixes(size, initial_depth, active_set_h);

initial.stop(); 

n_explorers = metrics[0]:int;
initial_tree_size = metrics[1];

metrics[0] = 0; //restarting for the parallel search_type
metrics[1] = 0;


///////////////////////////////////////////////////////////////////////////
//Distributed -- generating distributed Pool
///////////////////////////////////////////////////////////////////////////

writeln("It is distributed: ", distributed);

var D: domain(1) dmapped Block(boundingBox = {0..#n_explorers}) = {0..#n_explorers};
var dist_vector_of_tree_size_h: [D] c_uint;
var dist_sols_h: [D] c_uint;
var dist_active_set_h: [D] queens_node;

if distributed then {

	writeln("Starting copy and bulk transfer");

	bulktransfer.start();
	var centralized_active_set: [0..#n_explorers] queens_node;

	forall i in 0..#n_explorers do
    	centralized_active_set[i] = active_set_h[i];

   
    dist_active_set_h = centralized_active_set;
    bulktransfer.stop();
    writeln("Bulk transfer elapsed: ", bulktransfer.elapsed());
}



///////////////////////////////////////////////////////////////////////////
//Distributed wrapper
///////////////////////////////////////////////////////////////////////////

var DISTGPUWrapper = lambda (lo:int, hi: int, n_explorers: int) {
		//pointer arithmetics
		ref ldist_active_set_h= dist_active_set_h.localSlice(lo .. hi);
  		ref ldist_vector_of_tree_size_h = dist_vector_of_tree_size_h.localSlice(lo .. hi);
		ref ldist_sols_h = dist_sols_h.localSlice(lo .. hi);

  		GPU_call_cuda_queens(size, initial_depth, n_explorers:c_uint, 
			c_ptrTo(ldist_active_set_h),c_ptrTo(ldist_vector_of_tree_size_h), c_ptrTo(ldist_sols_h));
  	};



////////////////////////////////////////////////////////////////////
//// Search itself
////////////////////////////////////////////////////////////////////


	writeln("\nSize: ", size, " Survivors: ", n_explorers);        

	var num_gpus = GPU_device_count();

	writeln("Number of GPUs to use: ", num_gpus);   
	
	
	final.start();
	
	if distributed then{
		writeln("Distributed Search");
		forall i in GPU(D, DISTGPUWrapper, 0){
			;
		}
	}
	else{
		writeln("Single Locale");
		forall i in GPU(0..#(n_explorers:int), GPUWrapper, 0){
			;
		}
	}


////////////////////////////////////////////////////////////////////
//// Metrics reduction
////////////////////////////////////////////////////////////////////

	if distributed then{
		writeln("Distributed reduction.");

		var dist_redTree = (+ reduce dist_vector_of_tree_size_h):uint(64);
		var dist_redSol  = (+ reduce dist_sols_h):uint(64);
		final_tree_size = dist_redTree + initial_tree_size;
		final_sol = dist_redSol;

	}
	else{
		writeln("Single Locale reduction.");

		var redTree = (+ reduce vector_of_tree_size_h):uint(64);
		var redSol  = (+ reduce sols_h):uint(64);
		final_tree_size = redTree + initial_tree_size;
		final_sol = redSol;
	}
	
	

	final.stop();
	
	//var final_tree_size = initial_tree_size+metrics[2];
	writeln("Initial tree size: ", initial_tree_size);
	writeln("Final tree size: ", final_tree_size);
	writeln("Number of solutions: ", final_sol);
	if distributed then
		writeln("Elapsed time: ", final.elapsed()+initial.elapsed()+bulktransfer.elapsed());
	else
		writeln("Elapsed time: ", final.elapsed()+initial.elapsed());
//

