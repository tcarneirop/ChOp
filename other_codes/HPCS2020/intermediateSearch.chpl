        var maximum_number_prefixes: uint(64) = queens_get_number_prefixes(size,initial_depth);
        var maximum_number_prefixes_scnd_depth: uint(64) = queens_get_number_prefixes(size,second_depth);
        var set_size: uint(64) = maximum_number_prefixes_scnd_depth/maximum_number_prefixes;
    	var set_of_nodes: [0..set_size-1] queens_node; //
    	metrics += queens_improved_prefix_gen(size, initial_depth, second_depth, node, set_of_nodes);
        forall idx in dynamic(rangeDynamic, chunk) with (+ reduce metrics ) do {     metrics +=  queens_subtree_explorer(size,second_depth,set_of_nodes[idx:uint]);
        return metrics; //end of intermediate search, next line is the device search
        extern proc GPU_call_cuda_queens(size: uint(16), initial_depth:c_int, n_explorers:c_uint, root_prefixes_h: c_ptr(queens_node),vector_of_tree_size_h: c_ptr(c_uint), sols_h: c_ptr(c_uint),gpu_id:c_int): void;
        var vector_of_tree_size_h: [0..#initial_num_prefixes] c_uint;
        var sols_h: [0..#initial_num_prefixes] c_uint;
        forall gpu_id in 0..#num_gpus:c_int do{
                var gpu_load: c_uint = GPU_mlocale_get_gpu_load(new_num_prefixes:c_uint, gpu_id:c_int, num_gpus);
                var starting_position = GPU_mlocale_get_starting_point(new_num_prefixes:c_uint, gpu_id:c_uint, num_gpus:c_uint, cpu_load:c_uint);
                var sol_ptr : c_ptr(c_uint) = c_ptrTo(sols_h) + starting_position;
                var tree_ptr : c_ptr(c_uint) = c_ptrTo(vector_of_tree_size_h) + starting_position;
                var nodes_ptr : c_ptr(queens_node) = c_ptrTo(local_active_set) + starting_position;
                GPU_call_cuda_queens(size, depth, gpu_load:c_uint, nodes_ptr, tree_ptr, sol_ptr, gpu_id:c_int);}//end of gpu search
        var redTree = (+ reduce vector_of_tree_size_h):uint(64);
        var redSol  = (+ reduce sols_h):uint(64);
        return ((redSol,redTree)+metrics); //end of the device search, next line is the work distribution of the parallel search
forall n in distributed_active_set with (+ reduce metrics) do  {var m1 = queens_GPU_call_intermediate_search(size,initial_depth,second_depth,slchunk,n,tree_each_locale, GPU_id[here.id],CPUP);
        var GPU_id: [PrivateSpace] int;//From here, the code for launching the Mlocale mgpu application
        var metrics: (uint(64),uint(64)) = (0:uint(64),0:uint(64));
        var maximum_number_prefixes: uint(64) = queens_get_number_prefixes(size,initial_depth);
        var local_active_set: [0..maximum_number_prefixes-1] queens_node;
        metrics+= queens_node_generate_initial_prefixes(size, initial_depth, local_active_set );
        initial_num_prefixes = metrics[0];
        initial_tree_size = metrics[1];
        metrics[0] = 0; //restarting for the parallel search_type
        metrics[1] = 0;
        const Space = {0..(initial_num_prefixes-1):int}; //for distributing
        const D: domain(1) dmapped Block(boundingBox=Space) = Space; //1d block DISTRIBUTED
        var pgas_active_set: [D] queens_node; //1d block DISTRIBUTED
        var centralized_active_set: [Space] queens_node; 
        forall i in Space docentralized_active_set[i] = local_active_set[i:uint(64)];
        if(pgas){pgas_active_set =  centralized_active_set;
        if(num_gpus == 0) then{
            for loc in Locales do{
                on loc do{GPU_id[here.id] = GPU_device_count():int;                               
        else{
            if(GPU_device_count()<num_gpus) then
            for loc in Locales do{
                on loc do{GPU_id[here.id] = num_gpus:int;                               
       queens_mlocale_parameters_parser(size, scheduler, mode, mlsearch, initial_depth, second_depth, lchunk, mlchunk, slchunk, coordinated, centralized_active_set, Space, metrics,tree_each_locale,pgas,GPU_id,CPUP);