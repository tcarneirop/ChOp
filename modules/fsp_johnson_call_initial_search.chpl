
module fsp_johnson_call_initial_search{
    //use CPtr;
	use Time;
    use Math;

    use BlockDist;
    use CTypes;
    use CyclicDist;
    use PrivateDist;
    use VisualDebug;
    use DistributedIters;

	use fsp_aux;
    use fsp_constants;
    use fsp_aux_mlocale;
    use fsp_node_module;
    use fsp_johnson_aux_mlocale;
    use fsp_johnson_chpl_c_headers;
    use fsp_johnson_prefix_generation;


	proc fsp_johnson_call_initial_search(initial_depth: c_int = 4, upper_bound: c_int = _FSP_INF_,
        const scheduler: string = "dynamic", const chunk: int = 1, const num_threads: int,
        const profiler: bool = false, const atype: string = "none",const instance: c_short){

		print_locales_information();

		var initial,final, initialization,distribution: Timer;
		//FSP Variables
		var jobs: c_int;
    	var machines: c_int;
    	var times: c_ptr(c_int) = get_instance(machines,jobs, instance); //Get FSP problem
    	var local_times: [0..(machines*jobs)-1] c_int = [i in 0..(machines*jobs)-1] times[i];

        var global_ub: atomic c_int;

        const PrivateSpace: domain(1) dmapped Private();
        var set_of_atomics: [PrivateSpace] atomic c_int;

        var metrics: (uint(64),uint(64)) = (0:uint(64),0:uint(64));
        var initial_num_prefixes : uint(64) = 0;
        var initial_tree_size : uint(64) = 0;

    	var maximum_number_prefixes: uint(64) = fsp_get_number_prefixes(jobs,initial_depth);
        var local_active_set: [0..maximum_number_prefixes-1] fsp_node;


        //PROFILER
        if(profiler){
            startVdebug("Johnson"+scheduler+(instance:string)+(numLocales:string)+(machines:string)+(jobs:string));
            tagVdebug("read-only init");
            writeln("Starting profiler");
        }//end of profiler


        initialization.start();
        fsp_print_initial_info(scheduler,chunk,num_threads);
        fsp_all_locales_init_ub(upper_bound,global_ub,set_of_atomics);
        fsp_johnson_all_locales_get_instance(local_times, machines, jobs);
        fsp_johnson_all_locales_init_data(machines, jobs);
        initialization.stop();

        initial.start();
        //initial backtracking untill the cutoff depth
    	metrics += fsp_johnson_prefix_generation(machines,jobs,
    		upper_bound,times,initial_depth,local_active_set);
        initial.stop();

        initial_num_prefixes = metrics[0];
        initial_tree_size = metrics[1];

        metrics[0] = 0; //restarting for the parallel search_type
        metrics[1] = 0;

        //PROFILER
        if(profiler){
            tagVdebug("distribution");//profiler
        }

        writeln("####  PGAS Data Distribution  ####\n");
        distribution.start();
        //Distributed range
    	const Space = {0..(initial_num_prefixes-1):int}; //otherwise
		const D: domain(1) dmapped Block(boundingBox=Space) = Space; //1d block
    	//const D = Space dmapped Cyclic(startIdx=Space.low);
        var distributed_active_set: [D] fsp_node;

        //let's distribute the active set
        forall i in Space do
            distributed_active_set[i] = local_active_set[i:uint(64)];
        distribution.stop();


        if(profiler){
            stopVdebug();
        }


        fsp_call_initial_print_metrics(instance,  machines, jobs, initial,initialization,distribution,
            initial_tree_size, maximum_number_prefixes,initial_num_prefixes,
            upper_bound);

        distribution.clear();
        initial.clear();
        initialization.clear();



	}//distributed call


}
