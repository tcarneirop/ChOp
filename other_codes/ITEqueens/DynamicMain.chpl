
use SysCTypes;
//use GPUIterator;
use GPU_aux;
use CPtr;
use queens_prefix_generation;
use queens_tree_explorer;
use queens_node_module;
use queens_aux;
use Time;

use BlockDist;
use VisualDebug;
use CommDiagnostics;
use DistributedIters;

use ChapelLocks, DSIUtil;
use Time;

use SysCTypes;

config const size: uint(16) = 12;
config const initial_depth: c_int = 7;
config const distributed: bool = false;
config const CPUP: int = 100;

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
var max_number_prefixes: int = 75580635;
var active_set_h: [0..#max_number_prefixes] queens_node;

var n_explorers : int; 
	

//////////////////////////////////////////////////////////////////////////
//     Experimental dynamic iterator
/////////////////////////////////////////////////////////////////////////

const nGPUs = 1;

// serial iterator
iter GPU(D: domain,
         GPUWrapper,
         chunkSize: int = 4
    )
    where  isRectangularDom(D)
{
    for i in D do yield i;
}

// dynamic parallel standalone iterator
iter GPU(param tag: iterKind, D: domain, GPUWrapper,
         chunkSize: int = 4
    )
      where tag == iterKind.standalone
      && isRectangularDom(D)
{
    const numChunks: int;
    if D.idxType == uint(64) then
        numChunks = divceil(D.size, chunkSize:uint(64)): int;
    else
        numChunks = divceilpos(D.size:int(64), chunkSize): int;

    type rType=D.type;

    // We're going to have to densify at some point, might as well
    // do it early and make range slicing easier.
    const remain:rType=densify(D.dim(0),D.dim(0));

    const numCPUTasks = here.maxTaskPar;
    const numGPUTasks = nGPUs;
    const nTasks = numCPUTasks + numGPUTasks;

    var moreWork : atomic bool = true;
    var curChunkIdx : atomic int = 0;

    coforall tid in 0..#nTasks with (const in remain) {
        while moreWork.read() {
            const chunkIdx = curChunkIdx.fetchAdd(1);
            const low = chunkIdx * chunkSize; /* remain.low is 0, stride is 1 */
            const high: low.type;
            if chunkSize >= max(low.type) - low then
                high = max(low.type);
            else
                high = low + chunkSize-1;

            if chunkIdx >= numChunks {
                break;
            } else if high >= remain.high {
                moreWork.write(false);
            }
            const current:rType = remain(low .. high);
//            writeln("Parallel dynamic Iterator. Working at tid ", tid, " with range ", unDensify(current,D), " yielded as ", current);
            if (tid < numCPUTasks) {
                yield (current,);
            } else {
                GPUWrapper(low, high, high-low+1);
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////
//Single-locale wrapper
///////////////////////////////////////////////////////////////////////////

var GPUWrapper = lambda (lo:int, hi: int, n_explorers: int) {

	var sols_ptr : c_ptr(c_uint) = c_ptrTo(sols_h) + lo:c_uint;
	var tree_ptr : c_ptr(c_uint) = c_ptrTo(vector_of_tree_size_h) + lo:c_uint;
	var active_set_ptr : c_ptr(queens_node) = c_ptrTo(active_set_h) + lo:c_uint;

	GPU_call_cuda_queens(size, initial_depth, n_explorers:c_uint, 
		active_set_ptr,tree_ptr, sols_ptr);

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
		ref ldist_active_set_h = dist_active_set_h.localSlice(lo .. hi);
  		ref ldist_vector_of_tree_size_h = dist_vector_of_tree_size_h.localSlice(lo .. hi);
		ref ldist_sols_h = dist_sols_h.localSlice(lo .. hi);

  		GPU_call_cuda_queens(size, initial_depth, n_explorers:c_uint, 
			c_ptrTo(ldist_active_set_h),c_ptrTo(ldist_vector_of_tree_size_h), c_ptrTo(ldist_sols_h));
  	};



////////////////////////////////////////////////////////////////////
//// Search itself
////////////////////////////////////////////////////////////////////

  	proc test(i:int):void{
  		writeln("value: ", i, ". \n");
  	}

	writeln("\nSize: ", size, " Survivors: ", n_explorers);        

	var num_gpus = GPU_device_count();

	writeln("Number of GPUs to use: ", num_gpus);   
	
	
	final.start();
	
	if distributed then{
		writeln("Distributed Search");
		//forall i in GPU(D, DISTGPUWrapper, CPUP){
		//	(dist_sols_h[i],dist_vector_of_tree_size_h[i]) = 
		//		queens_subtree_explorer(size, initial_depth, dist_active_set_h[i]);
		//}
	}
	else{


		writeln("Single Locale");
		//forall i in GPU(0..#(n_explorers:int), GPUWrapper, CPUP){
		forall i in GPU(0..#(n_explorers:int), GPUWrapper, 50000){
			(sols_h[i],vector_of_tree_size_h[i]) = 
				queens_subtree_explorer(size, initial_depth, active_set_h[i]);
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

