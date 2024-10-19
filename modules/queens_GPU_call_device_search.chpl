
module queens_GPU_call_device_search{

	use queens_tree_exploration;
	use queens_node_module;
	use GPU_mlocale_utils;
	use CTypes;
	use GPU_aux;
	use DynamicIters;
	use Math;
	use Time;

	config param CPUGPUVerbose: bool = false;
	config param GPUAMD: bool = false;
	config param GPUCUDA: bool = false;


	//@todo: improve the CPU-GPU part, it is terrible...
	//--- I think this needs to be the most external function, so I can also add the chpl to the CPU-GPU part

	require "headers/CUDA_queens.h";

	extern proc  CUDA_call_queens(size: uint(16), initial_depth:c_int, n_explorers:c_uint,
			root_prefixes_h: c_ptr(queens_node),vector_of_tree_size_h: c_ptr(c_ulonglong),
			sols_h: c_ptr(c_ulonglong),gpu_id:c_int): void;
	

	require "headers/AMD_queens.h";

	extern proc AMD_call_queens(size: uint(16), initial_depth:c_int, n_explorers:c_uint,
			root_prefixes_h: c_ptr(queens_node),vector_of_tree_size_h: c_ptr(c_ulonglong),
			sols_h: c_ptr(c_ulonglong),gpu_id:c_int): void;
	


	proc queens_GPU_call_device_search(const num_gpus: c_int, const size: uint(16), const depth: c_int,
		ref local_active_set: [] queens_node, const initial_num_prefixes: uint(64),
		const CPUP: real, const chunk: int ): (uint(64), uint(64)){


		var vector_of_tree_size_h: [0..#initial_num_prefixes] c_ulonglong;
		var sols_h: [0..#initial_num_prefixes] c_ulonglong;


		//calculating the CPU load in terms of nodes
		var cpu_load: c_uint = (CPUP * initial_num_prefixes):c_uint;
		var new_num_prefixes: uint(64) = initial_num_prefixes - cpu_load:uint(64);
		var metrics: (uint(64),uint(64)) = (0:uint(64),0:uint(64));//

		
		//writeln("Locales: ",  Locales.size, " here.id: ",here.id, " here.gpus.size: ", here.gpus.size," Num gpus: ", num_gpus );
		

		coforall gpu_id in 0..#num_gpus:c_int do{

			///Maybe we need to modify it
			var gpu_load: c_uint = GPU_mlocale_get_gpu_load(new_num_prefixes:c_uint, gpu_id:c_int, num_gpus);

			var starting_position: c_uint = GPU_mlocale_get_starting_point(new_num_prefixes:c_uint,
				gpu_id:c_uint, num_gpus:c_uint, cpu_load:c_uint);

			var sol_ptr : c_ptr(c_ulonglong) = c_ptrTo(sols_h) + starting_position;
			var tree_ptr : c_ptr(c_ulonglong) = c_ptrTo(vector_of_tree_size_h) + starting_position;
			var nodes_ptr : c_ptr(queens_node) = c_ptrTo(local_active_set) + starting_position;
			var new_gpu_id: c_int;
			
			if Locales.size == 1 then new_gpu_id = gpu_id:c_int; else new_gpu_id = (here.id:c_int)%(here.gpus.size:c_int);
		
			//writeln("Locales: ",  Locales.size, " here.id: ",here.id, " here.gpus.size: ", here.gpus.size," GPU id: ", new_gpu_id, " Starting position: ", starting_position, " gpu load: ", gpu_load);
			
			if(GPUCUDA) then CUDA_call_queens(size, depth, gpu_load:c_uint,
				nodes_ptr, tree_ptr, sol_ptr, new_gpu_id);
			
			if(GPUAMD) then AMD_call_queens(size, depth, gpu_load:c_uint,
				nodes_ptr, tree_ptr, sol_ptr, new_gpu_id);

		}//end of gpu search



		if(CPUGPUVerbose){
			writeln("END OF THE SEARCH!");

		}

		var redTree = (+ reduce vector_of_tree_size_h):uint(64);
		var redSol  = (+ reduce sols_h):uint(64);

		return ((redSol,redTree)+metrics);

	}///

}
