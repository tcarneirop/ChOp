module fsp_simple_call_multilocale_search{
    //use CPtr;
	use Time;
    use Math;
    use List;
    use BlockDist;
    use CTypes;
    use CyclicDist;
    use PrivateDist;
    use VisualDebug;
    use statistics;
    use CommDiagnostics;

	use fsp_aux;
    use fsp_aux_mlocale;
    use fsp_node_module;
    use fsp_simple_aux_mlocale;
    use fsp_simple_chpl_c_headers;
    use fsp_simple_prefix_generation;
    use fsp_simple_mlocale_parameters_parser;
    use fsp_simple_improved_parameters_parser;


	proc fsp_simple_call_multilocale_search(initial_depth: c_int, second_depth: c_int, upper: c_int,
        const scheduler: string, const lchunk, const mlchunk, const slchunk,
        const coordinated: bool = false, const pgas: bool = false,
        const num_threads: int, const profiler: bool = false, const atype: string = "none", const instance: c_short,
        const mode: string = "improved", const verbose: bool = false, const diagnostics: bool = false): (real,real,real){

		print_locales_information();

		var initial,final, initialization,distribution: stopwatch;
        var return_initial: real;
        var return_final: real;
        var return_total: real;

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

        //var chunk_weight: [LocaleSpace] list(uint(64));

        //search metrics
        var metrics: (uint(64),uint(64)) = (0:uint(64),0:uint(64));
        var initial_num_prefixes : uint(64) = 0;
        var initial_tree_size : uint(64) = 0;

        //starting the search
    	var maximum_number_prefixes: uint(64) = fsp_get_number_prefixes(jobs,initial_depth);
        var local_active_set: [0..maximum_number_prefixes-1] fsp_node;

        //PROFILER
        if(profiler){
            startVdebug((numLocales:string)+"Simple"+"_"+"Coord"+(coordinated:string)+"_"
                +"PGAS"+(pgas:string)+"_"+scheduler+"_"+"Inst"+(instance:string)
                +"_"+(initial_depth:string)+(second_depth:string)+"_"
                +"chunks_"+(mlchunk:string)+"_"+(lchunk:string)+"_"+(slchunk:string));
            tagVdebug("read-only init");
            writeln("Starting profiler");
        }//end of profiler

        //initialization
        initialization.start();
        fsp_new_print_initial_info(initial_depth, second_depth, upper_bound,
        scheduler,lchunk, mlchunk, slchunk, coordinated,num_threads,
        atype,instance,mode, pgas);
        writeln("\n### 1 ###");
        fsp_all_locales_init_ub(upper_bound,global_ub,set_of_atomics);
        writeln("\n### 2 ###");
        statistics_all_locales_init_explored_tree(tree_each_locale);
        writeln("\n### 3 ###");
        fsp_simple_all_locales_get_instance(local_times, machines, jobs);
        writeln("\n### 4 ###");
        fsp_simple_all_locales_init_data(machines, jobs);
        writeln("\n### 5 ###");
        initialization.stop();


        initial.start();
        //initial search untill the cutoff depth
    	metrics += fsp_simple_prefix_generation(machines,jobs,
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

        if(diagnostics){
            writeln("\n### Starting communication counter ###");
            startCommDiagnostics();
        }

        //Distributer or centralized active set?

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
            pgas_active_set = centralized_active_set;
        }
        else{
            writeln("#####  Centralized active set #####\n");
            //forall i in Space do
            //    centralized_active_set[i] = local_active_set[i:uint(64)];
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
                    if pgas then
                        fsp_simple_mlocale_parameters_parser(atype, scheduler, machines,jobs, initial_depth,lchunk,
                            pgas_active_set, set_of_atomics, global_ub, Space, metrics);
                    else
                        fsp_simple_mlocale_parameters_parser(atype, scheduler, machines,jobs, initial_depth,lchunk,
                            centralized_active_set, set_of_atomics, global_ub, Space, metrics);

                }
                when "improved"{
                    if pgas then
                        fsp_simple_improved_mlocale_parameters_parser(atype, scheduler, machines,jobs, initial_depth,
                            second_depth,lchunk, mlchunk, slchunk, coordinated,pgas_active_set, set_of_atomics,
                            global_ub, Space, metrics,tree_each_locale,pgas);
                    else
                        fsp_simple_improved_mlocale_parameters_parser(atype, scheduler, machines,jobs, initial_depth,
                            second_depth,lchunk, mlchunk, slchunk, coordinated,centralized_active_set, set_of_atomics,
                            global_ub, Space, metrics,tree_each_locale,pgas);
                }
                otherwise{
                    halt("###### ERROR ######\n###### ERROR ######\n###### ERROR ######\n###### WRONG PARAMETERS ######");
                }
            }
        final.stop();

        ////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////

        if(profiler){
            stopVdebug();
        }

        fsp_mlocale_print_metrics(instance, machines, jobs, metrics, initial,initialization,distribution,final,
            initial_tree_size, maximum_number_prefixes,initial_num_prefixes,
            upper_bound, global_ub.read(),initial_depth,second_depth,tree_each_locale, mlchunk);

        if(diagnostics){
            writeln("### Stopping communication counter ###");
            stopCommDiagnostics();
            writeln("\n ### Communication results ### \n",getCommDiagnostics());
        }

        return_initial = initial.elapsed();
        return_final = final.elapsed();
        return_total = initial.elapsed()+final.elapsed();

        final.clear();
        initial.clear();
        initialization.clear();
        distribution.clear();

        return (return_initial,return_final,return_total);


	}//distributed call


}
