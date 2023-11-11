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



	/// Im not happy whith this file... I'll improve it.

	
	//require "headers/CUDA_queens.h";

	//extern proc  CUDA_call_queens(size: uint(16), initial_depth:c_int, n_explorers:c_uint,
	//		root_prefixes_h: c_ptr(queens_node),vector_of_tree_size_h: c_ptr(c_ulonglong),
	//		sols_h: c_ptr(c_ulonglong),gpu_id:c_int): void;

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

		//@question: should we use forall or coforall?

		if(cpu_load == 0){


			coforall gpu_id in 0..#num_gpus:c_int do{

					var gpu_load: c_uint = GPU_mlocale_get_gpu_load(new_num_prefixes:c_uint, gpu_id:c_int, num_gpus);

					var starting_position = GPU_mlocale_get_starting_point(new_num_prefixes:c_uint,
						gpu_id:c_uint, num_gpus:c_uint, cpu_load:c_uint);

					var sol_ptr : c_ptr(c_ulonglong) = c_ptrTo(sols_h) + starting_position;
					var tree_ptr : c_ptr(c_ulonglong) = c_ptrTo(vector_of_tree_size_h) + starting_position;
					var nodes_ptr : c_ptr(queens_node) = c_ptrTo(local_active_set) + starting_position;

					if(CPUGPUVerbose) then
						writeln("GPU id: ", gpu_id, " Starting position: ", starting_position, " gpu load: ", gpu_load);
					
					if(GPUCUDA) then CUDA_call_queens(size, depth, gpu_load:c_uint,
						nodes_ptr, tree_ptr, sol_ptr, gpu_id:c_int);
					
					if(GPUAMD) then AMD_call_queens(size, depth, gpu_load:c_uint,
						nodes_ptr, tree_ptr, sol_ptr, gpu_id:c_int);

				}//end of gpu search

		 }
		 else{

		 	cobegin with (ref metrics){

		 		{/////
		 			//if(CPUGPUVerbose){//use this variable to see the debug messages
		 				writeln("CPUP: ", CPUP);
		 				writeln("Going on CPU");
		 			//}

		 			forall idx in dynamic(0..(cpu_load:int), chunk) with (+ reduce metrics ) do {
		 				metrics +=  queens_subtree_explorer(size,depth,local_active_set[idx:uint]);
		 			}

		 			if(CPUGPUVerbose){
		 				writeln("End of the CPU search.");
		 			}

		 		}////


		 		{
		 		//two blocks, two tasks
		 		//Other task for the GPUs

		 			if(CPUGPUVerbose){

		 				writeln("Remaining nodes: ", new_num_prefixes);
		 				writeln("Starting position for GPU: ", cpu_load+1);
		 				writeln("Going on GPU");

		 			}

		 			coforall gpu_id in 0..#num_gpus:c_int do{

		 				var gpu_load: c_uint = GPU_mlocale_get_gpu_load(new_num_prefixes:c_uint, gpu_id:c_int, num_gpus);

		 				var starting_position = GPU_mlocale_get_starting_point(new_num_prefixes:c_uint,
		 					gpu_id:c_uint, num_gpus:c_uint, cpu_load:c_uint);

		 				var sol_ptr : c_ptr(c_ulonglong) = c_ptrTo(sols_h) + starting_position;
		 				var tree_ptr : c_ptr(c_ulonglong) = c_ptrTo(vector_of_tree_size_h) + starting_position;
		 				var nodes_ptr : c_ptr(queens_node) = c_ptrTo(local_active_set) + starting_position;

		 				if(CPUGPUVerbose) then writeln("GPU id: ", gpu_id, " Starting position: ", starting_position, " gpu load: ", gpu_load);
		 				
		 				if(GPUCUDA) then CUDA_call_queens(size, depth, gpu_load:c_uint,
							nodes_ptr, tree_ptr, sol_ptr, gpu_id:c_int);
					
						if(GPUAMD) then AMD_call_queens(size, depth, gpu_load:c_uint,
							nodes_ptr, tree_ptr, sol_ptr, gpu_id:c_int);

		 			}//end of gpu search


		 			if(CPUGPUVerbose){
		 				writeln("end of the GPU search");
		 			}


		 		}
		 	}//end of cobegin
		 }//else


		if(CPUGPUVerbose){
			writeln("END OF THE SEARCH!");

		}

		var redTree = (+ reduce vector_of_tree_size_h):uint(64);
		var redSol  = (+ reduce sols_h):uint(64);

		return ((redSol,redTree)+metrics);

	}///




}
