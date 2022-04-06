module queens_GPU_call_device_search{

	use queens_tree_exploration;
	use queens_node_module;
	use GPU_mlocale_utils;
	use SysCTypes;
	use GPU_aux;
	use DynamicIters;
	use Math;
	use CPtr;
	use DateTime;

	config param CPUGPUVerbose: bool = false;

	require "headers/GPU_queens.h";


	extern proc GPU_call_cuda_queens(size: uint(16), initial_depth:c_int, n_explorers:c_uint,
		root_prefixes_h: c_ptr(queens_node),vector_of_tree_size_h: c_ptr(c_ulonglong),
		sols_h: c_ptr(c_ulonglong),gpu_id:c_int): void;


	proc queens_GPU_call_device_search(const num_gpus: c_int, const size: uint(16), const depth: c_int,
		ref local_active_set: [] queens_node, const initial_num_prefixes: uint(64)): (uint(64), uint(64)){


		var vector_of_tree_size_h: [0..#initial_num_prefixes] c_ulonglong;
		var sols_h: [0..#initial_num_prefixes] c_ulonglong;


		//calculating the CPU load in terms of nodes
		var new_num_prefixes: uint(64) = initial_num_prefixes;
		var metrics: (uint(64),uint(64)) = (0:uint(64),0:uint(64));//

		//@question: should we use forall or coforall?

		forall gpu_id in 0..#num_gpus:c_int do{

				var gpu_load: c_uint = GPU_mlocale_get_gpu_load(new_num_prefixes:c_uint, gpu_id:c_int, num_gpus);

				var starting_position = GPU_mlocale_get_starting_point(new_num_prefixes:c_uint,
					gpu_id:c_uint, num_gpus:c_uint, 0:c_uint);

				var sol_ptr : c_ptr(c_ulonglong) = c_ptrTo(sols_h) + starting_position;
				var tree_ptr : c_ptr(c_ulonglong) = c_ptrTo(vector_of_tree_size_h) + starting_position;
				var nodes_ptr : c_ptr(queens_node) = c_ptrTo(local_active_set) + starting_position;

				if(CPUGPUVerbose) then
					writeln("GPU id: ", gpu_id, " Starting position: ", starting_position, " gpu load: ", gpu_load);

				GPU_call_cuda_queens(size, depth, gpu_load:c_uint,
					nodes_ptr, tree_ptr, sol_ptr, gpu_id:c_int);

			}//end of gpu search

		var redTree = (+ reduce vector_of_tree_size_h):uint(64);
		var redSol  = (+ reduce sols_h):uint(64);

		return ((redSol,redTree)+metrics);

	}///
}
