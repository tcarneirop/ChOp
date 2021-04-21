module queens_GPU_call_intermediate_search{

	use queens_node_module;
	use queens_prefix_generation;
	use queens_aux;
	use queens_tree_exploration;
	use queens_GPU_call_device_search;

	use queens_aux;
	use DynamicIters;
	use SysCTypes;

	extern proc GPU_call_cuda_queens(size: uint(16), initial_depth:c_int, n_explorers:c_uint, 
		root_prefixes_h: c_ptr(queens_node),vector_of_tree_size_h: c_ptr(c_int), 
		sols_h: c_ptr(c_int), gpu_id:c_int): void;

	proc queens_GPU_call_intermediate_search(const size: uint(16), const initial_depth: c_int, 
		const second_depth: c_int, const chunk: int, ref node: queens_node,
		ref tree_each_locale: [] uint(64), const GPU: int, const CPUP: real):(uint(64),uint(64)){

		var maximum_number_prefixes: uint(64) = queens_get_number_prefixes(size,initial_depth);//
		var maximum_number_prefixes_scnd_depth: uint(64) = queens_get_number_prefixes(size,second_depth);//
		var set_size: uint(64) = maximum_number_prefixes_scnd_depth/maximum_number_prefixes;//

		var set_of_nodes: [0..set_size-1] queens_node; //
		//metrics
		var metrics: (uint(64),uint(64)) = (0:uint(64),0:uint(64));//
		var initial_num_prefixes : uint(64) = 0;//
		var initial_tree_size : uint(64) = 0;//

		metrics += queens_improved_prefix_gen(size, initial_depth, second_depth, node, set_of_nodes);//
		
		initial_num_prefixes = metrics[0];//
		metrics[0] = 0; //restarting for the parallel search_type//


		metrics+= queens_GPU_call_device_search(GPU:c_int, size, second_depth, set_of_nodes,
		 initial_num_prefixes, CPUP, chunk);
		
		//}//end of GPU portion of the code
		//else{//if it is a  CPU locale

		//	forall idx in dynamic(0..#initial_num_prefixes, chunk) with (+ reduce metrics ) do {     
		//		metrics +=  queens_subtree_explorer(size,second_depth,set_of_nodes[idx:uint]);
		//	}//search

		//}///else
	
		tree_each_locale[here.id] += metrics[1]; //for load statistics

		return metrics;	
	}

} //if we forget to put the }, we just get a modules/queens_GPU_call_intermediate_search.chpl:50: syntax error
