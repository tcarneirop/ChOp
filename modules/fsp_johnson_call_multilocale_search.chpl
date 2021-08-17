
module fsp_johnson_call_multilocale_search{
	
	use Time;
    use Math;
    use List;
    use CPtr;
    use BlockDist;
    use SysCTypes;
    use CyclicDist;
    use PrivateDist;
    use VisualDebug;
    use DistributedIters;
    use statistics;
    use CommDiagnostics;

	use fsp_aux;
    use fsp_aux_mlocale;
    use fsp_node_module;
    use fsp_johnson_aux_mlocale;
    use fsp_johnson_chpl_c_headers;
    use fsp_johnson_prefix_generation;
   // use fsp_johnson_mlocale_parameters_parser;
    use fsp_johnson_improved_parameters_parser;

    //pro fsp_johsin(ref struct)
	proc fsp_johnson_call_multilocale_search(initial_depth: c_int, second_depth: c_int, upper: c_int, 
        const scheduler: string, const lchunk, const mlchunk, const slchunk,
        const coordinated: bool = false, const pgas: bool = false,
        const num_threads: int, const profiler: bool = false, const atype: string = "none", const instance: c_short, 
        const mode: string, const verbose: bool = false
        ){


		print_locales_information();

		var initial,final, initialization,distribution: Timer;

		//FSP Variables
		var jobs: c_int;
    	var machines: c_int;
    	var times: c_ptr(c_int) = get_instance(machines,jobs, instance); //Get FSP problem
    	var local_times: [0..(machines*jobs)-1] c_int = [i in 0..(machines*jobs)-1] times[i];

        //ub initialization
        var global_ub: atomic c_int;
        var upper_bound: c_int = fsp_get_upper_bound(upper,instance); 
   

        //one atomic for each node
        const PrivateSpace: domain(1) dmapped Private();
        var set_of_atomics: [PrivateSpace] atomic c_int;
        var tree_each_locale: [PrivateSpace] uint(64);

       // var chunk_weight: [PrivateSpace] list(uint(64), parSafe = true);
        

        //search metrics
        var metrics: (uint(64),uint(64)) = (0:uint(64),0:uint(64));
        var initial_num_prefixes : uint(64) = 0;
        var initial_tree_size : uint(64) = 0;

        //starting the search
    	var maximum_number_prefixes: uint(64) = fsp_get_number_prefixes(jobs,initial_depth);
        var local_active_set: [0..maximum_number_prefixes-1] fsp_node;


        //PROFILER
        if(profiler){
            startVdebug((numLocales:string)+"Johnson"+"_"+"Coord"+(coordinated:string)+"_"
                +"PGAS"+(pgas:string)+"_"+scheduler+"_"+"Inst"+(instance:string)
                +"_"+(initial_depth:string)+(second_depth:string)+"_"
                +"chunks_"+(mlchunk:string)+"_"+(lchunk:string)+"_"+(slchunk:string));
            tagVdebug("read-only init");
            writeln("Starting profiler");
        }//end of profiler

        //initialization timer
        initialization.start();

        fsp_new_print_initial_info(initial_depth, second_depth, upper_bound, 
        scheduler,lchunk, mlchunk,slchunk, coordinated,num_threads, 
        atype,instance,mode, pgas);
        fsp_all_locales_init_ub(upper_bound,global_ub,set_of_atomics);
        statistics_all_locales_init_explored_tree(tree_each_locale);
        fsp_johnson_all_locales_get_instance(local_times, machines, jobs);
        fsp_johnson_all_locales_init_data(machines, jobs);

        initialization.stop();


        //initial search untill the cutoff depth
        initial.start();
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
        
        distribution.start();

    
        fsp_is_aset_empty(initial_num_prefixes,initial_tree_size);

        //Distributer or centralized active set?

        if(verbose){
            writeln("\n### Starting communication counter ###");
            startCommDiagnostics();
        }

    	const Space = {0..(initial_num_prefixes-1):int}; //for distributing
        const D: domain(1) dmapped Block(boundingBox=Space) = Space; //1d block DISTRIBUTED
        var pgas_active_set: [D] fsp_node; //1d block DISTRIBUTED
        var centralized_active_set: [Space] fsp_node; //on node 0

        //let's distribute the active set 
        writeln("####  initialization of the Active Set  ####\n");
        forall i in Space do
        	centralized_active_set[i] = local_active_set[i:uint(64)];
        if(pgas){
            writeln("#####  PGAS-based active set #####\n");
            //I changed here. Now, the active set is initilized with = operator.
            pgas_active_set = centralized_active_set;
        }
        else{
        	//Im going to remove here, because ill use the = operator for PGAS-based data distribution.
        	//So, im removing the forall and using for distributed and pgas.
            writeln("#####  Centralized active set #####\n");
        }

        distribution.stop();


        //PROFILER
        if(profiler){
            tagVdebug(scheduler);//profiler
        }

        ////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////
        
        final.start();

        writeln("#### Nodes to explore: ", initial_num_prefixes);

        select mode {
            
            when "mlocale"{
                 //if pgas then 
                   // fsp_johnson_mlocale_parameters_parser(atype, scheduler, machines,jobs, initial_depth,lchunk,
                   //      pgas_active_set, set_of_atomics, global_ub, Space, metrics, local_timer);
                 //else
                    // fsp_johnson_mlocale_parameters_parser(atype, scheduler, machines,jobs, initial_depth,lchunk,
                    //     centralized_active_set, set_of_atomics, global_ub, Space, metrics,local_timer);
            }//end of selection
            when "improved"{
                if pgas then 
                    fsp_johnson_improved_parameters_parser(atype, scheduler, machines,jobs, initial_depth,
                        second_depth,lchunk, mlchunk, slchunk,coordinated,pgas_active_set, set_of_atomics, 
                        global_ub, Space, metrics,tree_each_locale, pgas);
                else
                    fsp_johnson_improved_parameters_parser(atype, scheduler, machines,jobs, initial_depth,
                        second_depth,lchunk, mlchunk, slchunk, coordinated,centralized_active_set, set_of_atomics, 
                        global_ub, Space, metrics,tree_each_locale, pgas);

            }//end of selection  
            otherwise{
                writeln("#### Wrong parameters ####");
            }
        }//mlocalemode

        final.stop();

        ////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////
        
        if(profiler){
            stopVdebug();
        }
       
        fsp_mlocale_print_metrics( instance, machines, jobs, metrics, initial,initialization,distribution,final, initial_tree_size, 
            maximum_number_prefixes,initial_num_prefixes, upper_bound, global_ub.read(), 
            initial_depth, second_depth,tree_each_locale, mlchunk);        

        if(verbose){
            writeln("### Stopping communication counter ###");
            stopCommDiagnostics();
            writeln("\n ### Communication results ### \n",getCommDiagnostics());
        }
        
        final.clear();
        initial.clear();
        initialization.clear();
        distribution.clear();

	}//distributed call

}
