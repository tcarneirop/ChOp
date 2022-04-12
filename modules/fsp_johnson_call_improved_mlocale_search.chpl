module fsp_johnson_call_improved_mlocale_search{


	use fsp_johnson_multilocale_node_explorer;
	use fsp_johnson_improved_prefix_gen;
	use fsp_johnson_chpl_c_headers;

	use fsp_node_module;
	use fsp_aux;

    use DynamicIters;
	use CTypes;
    use List;

	proc fsp_johnson_call_improved_mlocale_search(const machines: c_int,const jobs: c_int,
    	const initial_depth: c_int ,const second_depth: c_int,
    	const chunk: int, ref node: fsp_node,global_ub: c_int,
        ref tree_each_locale: [] uint(64)):(uint(64),uint(64)){


		//FSP Variables
        var maximum_number_prefixes: uint(64) = fsp_get_number_prefixes(jobs,initial_depth);
        var maximum_number_prefixes_scnd_depth: uint(64) = fsp_get_number_prefixes(jobs,second_depth);
        var set_size: uint(64) = maximum_number_prefixes_scnd_depth/maximum_number_prefixes;

    	var set_of_nodes: [0..set_size-1] fsp_node;

    	//metrics
    	var metrics: (uint(64),uint(64)) = (0:uint(64),0:uint(64));
        var initial_num_prefixes : uint(64) = 0;

    	metrics += fsp_johnson_improved_prefix_gen(machines,jobs, initial_depth, second_depth, node,
    		set_of_nodes, global_ub);


        initial_num_prefixes = metrics[0];
        metrics[0] = 0; //restarting for the parallel search_type


        var aux: int = initial_num_prefixes: int;
        var rangeDynamic: range = 0..aux-1;

        forall idx in dynamic(rangeDynamic, chunk) with (+ reduce metrics ) do {
        	metrics +=  fsp_johnson_mlocale_node_explorer(machines,jobs,second_depth,
                set_of_nodes[idx:uint], global_ub);
        }//search

        tree_each_locale[here.id] += metrics[1];

		return metrics;

	}//proc


}//module
