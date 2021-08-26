
module fsp_simple_call_mcore_search{

	use fsp_simple_chpl_c_headers;
	use fsp_constants;
    use fsp_node_module;
	use fsp_simple_prefix_generation;
    use fsp_simple_node_explorer;
    use fsp_aux;
    use concurrency;	
    use DynamicIters;
	use SysCTypes;
	use Time;
    use CPtr;
    
    config param methodStealing = Method.WholeTail;

	proc fsp_simple_call_multicore_search(initial_depth: c_int, upper: c_int, 
        const scheduler: string, const chunk: int, const num_threads, const instance: c_short){

		var initial,final: Timer;

		//FSP Variables
		var jobs: c_int;
    	var machines: c_int;
    	var times: c_ptr(c_int) = get_instance(machines,jobs,instance); //Get FSP problem

    	initial.start();
        var maximum_number_prefixes: uint(64) = fsp_get_number_prefixes(jobs,initial_depth);
    	var set_of_nodes: [0..maximum_number_prefixes-1] fsp_node;

    	//metrics
    	var metrics: (uint(64),uint(64)) = (0:uint(64),0:uint(64));
        var initial_num_prefixes : uint(64) = 0;
        var initial_tree_size : uint(64) = 0;

        //atomic and global upper bound
        var upper_bound: c_int = fsp_get_upper_bound(upper,instance); 
        var global_ub: atomic c_int; 
        global_ub.write(upper_bound);

        //start read-only data
        remplirTempsArriverDepart(minTempsArr_s, minTempsDep_s, 
            machines,jobs,times);
    	//initial search. Maybe, we need to parallelize it
    	metrics += fsp_simple_prefix_generation(machines,jobs,
    		upper_bound,times,initial_depth,set_of_nodes);
    	
        initial.stop(); 

        initial_num_prefixes = metrics[0];
        initial_tree_size = metrics[1];

        metrics[0] = 0; //restarting for the parallel search_type
        metrics[1] = 0;

        var aux: int = initial_num_prefixes: int;
        var rangeDynamic: range = 0..aux-1;
        
        fsp_print_mcore_initial_info(initial_depth, upper_bound, scheduler, chunk,num_threads,instance);

        final.start();//calculating time
        select scheduler{
            when "static" {
                 forall idx in 0..initial_num_prefixes-1 with (+ reduce metrics) do {   
                    metrics +=  fsp_simple_node_explorer(machines,jobs,global_ub,times,initial_depth,
                        set_of_nodes[idx:uint]); 
                 }//search
            }//static
            when "dynamic" {            
                forall idx in dynamic(rangeDynamic, chunk, num_threads) with (+ reduce metrics ) do {
                    metrics +=  fsp_simple_node_explorer(machines,jobs,global_ub,times,initial_depth,
                        set_of_nodes[idx:uint]);
                }//search
            }//dynamic
            when "guided" {
                forall idx in guided(rangeDynamic,num_threads) with (+ reduce metrics ) do {
                    metrics +=  fsp_simple_node_explorer(machines,jobs,global_ub,times,initial_depth,
                            set_of_nodes[idx:uint]); 
                }//search
            }//guided
            when "stealing" {
                forall idx in adaptive(rangeDynamic, num_threads) with (+ reduce metrics ) do {
                    metrics +=  fsp_simple_node_explorer(machines,jobs,global_ub,times,initial_depth,
                        set_of_nodes[idx:uint]);
                }//work stealing
            }//search
            otherwise{
                writeln("\n\n ###### error ######\n\n ###### error ######\n\n ###### error ###### ");
            }
        }//select
        final.stop();
        
        fsp_print_metrics( machines, jobs, metrics, initial,final, initial_tree_size, 
            maximum_number_prefixes,initial_num_prefixes, upper_bound, global_ub);        

        final.clear();
        initial.clear();
    	
	}//Call mcore search

	

}//module