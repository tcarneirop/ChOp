module queens_GPU_call_device_search{

	use queens_node_module;
	use GPU_mlocale_utils;
	use SysCTypes;
	use GPU_aux;

	require "headers/GPU_queens.h";


	//extern proc GPU_call_cuda_queens(size: uint(16), initial_depth:c_int, n_explorers:c_uint, 
	//	root_prefixes_h: c_ptr(queens_node),vector_of_tree_size_h: c_ptr(c_uint), 
	//	sols_h: c_ptr(c_uint),gpu_id:c_int): void;


	extern proc GPU_call_cuda_queens(size: uint(16), initial_depth:c_int, n_explorers:c_uint, 
		root_prefixes_h: c_ptr(queens_node),vector_of_tree_size_h: c_ptr(c_uint), 
		sols_h: c_ptr(c_uint)): void;


	config param size: 


	var vector_of_tree_size_h: [0..#10000] c_uint;
	var sols_h: [0..#10000] c_uint;

	forall i in GPU(1..n, GPUWrapper, CPUPercent) {
 	 // CPU code
  		A(i) = B(i);
	}


	//I'm not using int -- what should I do?
	var GPUWrapper = lambda (lo:int, hi: int, num_explorers: int) {
  		GPU_call_cuda_queens(size, initial_depth, n_explorers:c_uint, 
		root_prefixes_h,vector_of_tree_size_h, 
		sols_h);
  	};
	//var CPUPercent = 50; // CPUPercent is optional


	proc queens_GPU_iterator_device_search(const num_gpus: c_int, const size: uint(16), const depth: c_int, 
			ref local_active_set: [] queens_node, const initial_num_prefixes: uint(64)): (uint(64), uint(64)){

		


		var queensGPUWrapper = lambda (){ //@QUESTION: what should I pass here?
			GPU_call_cuda_queens(size, depth, gpu_load:c_uint, 
				local_active_set, vector_of_tree_size_h, sols_h); 
				//@QUESTION: what should I do concerning the pointer arithmetics?
				//@QUESTION: what should I do with the GPU ID?
		};


		forall i in GPU(0..#size, queensGPUWrapper, CPUPercent) { //@QUESTION: is 0..#size lo..hi?
  			// CPU code //@QUESTION:i did not get -- should I write CPU code by hand?
  			A(i) = B(i); 
  			//@QUESTION: is the CPU code automatically generated based on the GPU kernel?
		}



		var redTree = (+ reduce vector_of_tree_size_h):uint(64);
		var redSol  = (+ reduce sols_h):uint(64);
	

		return (redSol,redTree);

	}




	proc queens_GPU_call_device_search(const num_gpus: c_int, const size: uint(16), const depth: c_int, 
		ref local_active_set: [] queens_node, const initial_num_prefixes: uint(64)): (uint(64), uint(64)){

		var vector_of_tree_size_h: [0..#initial_num_prefixes] c_uint;
		var sols_h: [0..#initial_num_prefixes] c_uint;


		//the gpu search comes here
		coforall gpu_id in 0..#num_gpus do{

			var gpu_load: c_uint = GPU_mlocale_get_gpu_load(initial_num_prefixes:c_uint, gpu_id:c_int,  num_gpus);

			var starting_position = GPU_mlocale_get_starting_point(initial_num_prefixes:c_uint, gpu_id:c_int,  num_gpus);

			var sol_ptr : c_ptr(c_uint) = c_ptrTo(sols_h) + starting_position;
			var tree_ptr : c_ptr(c_uint) = c_ptrTo(vector_of_tree_size_h) + starting_position;
			var nodes_ptr : c_ptr(queens_node) = c_ptrTo(local_active_set) + starting_position;

			//writeln(here.id, " ", gpu_id, " ", starting_position, " ", gpu_load);
			GPU_call_cuda_queens(size, depth, gpu_load:c_uint, 
				nodes_ptr, tree_ptr, sol_ptr, gpu_id:c_int);
			
		}//end of gpu search

		var redTree = (+ reduce vector_of_tree_size_h):uint(64);
		var redSol  = (+ reduce sols_h):uint(64);
	

		return (redSol,redTree);

	}///
}