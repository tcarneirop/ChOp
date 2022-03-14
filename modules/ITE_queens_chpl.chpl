module ITE_queens_chpl{

		
	use SysCTypes;
	//use GPUIterator;
	use GPU_aux;
	use CPtr;
	use queens_prefix_generation;
	use queens_tree_exploration;
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
	config const CPUP: int = 100;

	///////////////////////////////////////////////////////////////////////////
	//C-Interoperability
	///////////////////////////////////////////////////////////////////////////

	require "headers/GPU_queens.h";


	extern proc GPU_call_cuda_queens(size: uint(16), initial_depth:c_int, n_explorers:c_uint, 
			active_set_h: c_ptr(queens_node),vector_of_tree_size_h: c_ptr(c_longlong), 
			sols_h: c_ptr(c_longlong)): void;


	///////////////////////////////////////////////////////////////////////////
	//CUDA Vectors
	///////////////////////////////////////////////////////////////////////////

	var vector_of_tree_size_h: [0..#75580635] c_longlong;
	var sols_h: [0..#75580635] c_longlong;

	//var active_set_h: [0..#] queens_node;
	var max_number_prefixes: int = 75580635;
	var active_set_h: [0..#max_number_prefixes] queens_node;

	var n_explorers : int; 
		

	///////////////////////////////////////////////////////////////////////////
	//Single-locale wrapper
	///////////////////////////////////////////////////////////////////////////

	var GPUWrapper = lambda (lo:int, hi: int, n_explorers: int) {

		var sols_ptr : c_ptr(c_longlong) = c_ptrTo(sols_h) + lo:c_longlong;
		var tree_ptr : c_ptr(c_longlong) = c_ptrTo(vector_of_tree_size_h) + lo:c_longlong;
		var active_set_ptr : c_ptr(queens_node) = c_ptrTo(active_set_h) + lo:c_uint;

		GPU_call_cuda_queens(size, initial_depth, n_explorers:c_uint, 
			active_set_ptr,tree_ptr, sols_ptr);

	};

}