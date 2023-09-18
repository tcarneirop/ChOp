module queens_call_multilocale_search{


    //use queens_constants;
    use queens_node_module;
    use queens_prefix_generation;
    use queens_mlocale_parameters_parser;
    use queens_aux;
    use GPU_mlocale_utils;
    use GPU_aux;
    use Time;
    use statistics;

    use BlockDist;
    use CTypes;
    use PrivateDist;
    use CyclicDist;
    use VisualDebug;
    use DistributedIters;
    use CommDiagnostics;

    proc queens_call_multilocale_search(const size: uint(16), const initial_depth: c_int,
        const second_depth: c_int, const scheduler: string, const mode: string, const mlsearch:string,
        const lchunk: int,
        const mlchunk: int, const slchunk: int,
        const coordinated: bool = false, const pgas: bool = false,
        const num_threads: int, const profiler: bool = false,
        const verbose: bool = false,
        const CPUP: real, const num_gpus: c_int):(real,real,real){


        if verbose then queens_print_locales_information();
        queens_print_mlocale_initial_info(size, initial_depth, second_depth, scheduler,lchunk, mlchunk, slchunk,
            coordinated,num_threads, mode,pgas);

        var initial_num_prefixes : uint(64);
        var initial_tree_size : uint(64) = 0;
        var number_of_solutions: uint(64) = 0;
        var final_tree_size: uint(64) = 0;
        var parallel_tree_size: uint(64) = 0;

        var initial, final, distribution: stopwatch;
        var return_initial: real;
        var return_final: real;
        var return_total: real;


        const PrivateSpace: domain(1) dmapped Private();
        var tree_each_locale: [PrivateSpace] uint(64);
        var GPU_id: [PrivateSpace] int;
        var Locale_role: [PrivateSpace] int;


        statistics_all_locales_init_explored_tree(tree_each_locale);

        //search metrics
        var metrics: (uint(64),uint(64)) = (0:uint(64),0:uint(64));

        //PROFILER
        if(profiler){
            startVdebug("QUEENS"+(numLocales:string)+"_"+"Coord"+(coordinated:string)+"_"
                +"PGAS"+(pgas:string)+"_"+scheduler+"_"+"_"+(initial_depth:string)+(second_depth:string)+"_"
                +"chunks_"+(mlchunk:string)+"_"+(lchunk:string)+"_"+(slchunk:string));
            tagVdebug("read-only init");
            writeln("Starting profiler");
        }//end of profiler

        //starting the search
        initial.start();
        var maximum_number_prefixes: uint(64) = queens_get_number_prefixes(size,initial_depth);
        var local_active_set: [0..maximum_number_prefixes-1] queens_node;
        metrics+= queens_node_generate_initial_prefixes(size, initial_depth, local_active_set );
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

        if(verbose){
            writeln("\n### Starting communication counter ###");
            startCommDiagnostics();
        }

        //Distributer or centralized active set?
        const Space = {0..(initial_num_prefixes-1):int}; //for distributing
        const D: domain(1) dmapped Block(boundingBox=Space) = Space; //1d block DISTRIBUTED
        var pgas_active_set: [D] queens_node; //1d block DISTRIBUTED
        var centralized_active_set: [Space] queens_node; //on node 0

        //let's distribute the active set
        writeln("####  initialization of the Active Set  ####");
        forall i in Space do
            centralized_active_set[i] = local_active_set[i:uint(64)];
        if(pgas){
            writeln("#####  PGAS-based active set #####");
            //forall i in Space do
            pgas_active_set =  centralized_active_set;
        }
        else{
            writeln("#####  Centralized active set #####");
            //forall i in Space do
            //    centralized_active_set[i] = local_active_set[i:uint(64)];
        }
        distribution.stop();


        //for mlmgpu and gpucpu
        //When one wants to use less GPUs than the available number
        //if(num_gpus == 0) then{
          //  if verbose then writeln("\n # The total number of GPU is going to be used (DEFAULT): ", GPU_device_count() ,". #");
            //for loc in Locales do{
              //  on loc do{
              //      GPU_id[here.id] = GPU_device_count():int;
            //    }//on loc
          //  }///fors
        //}
        //else{
           // if( GPU_device_count()< num_gpus) then
            //    halt("###### ERROR ######\n###### ERROR ######\n###### ERROR ######\n###### NUMBER OF AVAILABLE DEVICES < NUM_GPUS ######");
            //says that each locale must use x gpus
            writeln("\n # Number of GPUs : ", num_gpus ," #");
            for loc in Locales do{
                on loc do{
                    GPU_id[here.id] = num_gpus;
                }//on loc
            }///for
       // }



        //@todo @todo
        //@todo: going to perform the check here.
        //GPU_mlocale_number_locales_check(mlsearch, coordinated:int);



        //PROFILER
        if(profiler){
            tagVdebug(scheduler);//profiler
        }
        ////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////
        final.start();
        writeln("\n\n\n#### Nodes to explore: ", initial_num_prefixes);
        //Launching the search
        //(i)
        if pgas then
            queens_mlocale_parameters_parser(size, scheduler, mode, mlsearch,initial_depth,
                second_depth,lchunk, mlchunk, slchunk,coordinated,pgas_active_set,
                Space, metrics,tree_each_locale,pgas,GPU_id, CPUP);
        else
            queens_mlocale_parameters_parser(size, scheduler, mode, mlsearch, initial_depth,
                second_depth, lchunk, mlchunk, slchunk, coordinated, centralized_active_set,
                Space, metrics,tree_each_locale,pgas,GPU_id,CPUP);

        final.stop();

        ////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////

        if(profiler){
            stopVdebug();
        }

        queens_mlocale_print_metrics(size,metrics, initial, distribution,final,
            initial_tree_size, maximum_number_prefixes, initial_num_prefixes, initial_depth,second_depth,tree_each_locale);

        if(verbose){
            writeln("### Stopping communication counter ###");
            stopCommDiagnostics();
            writeln("\n ### Communication results ### \n",getCommDiagnostics());
        }

        return_initial = initial.elapsed();
        return_final = final.elapsed();
        return_total = initial.elapsed()+final.elapsed();

        final.clear();
        initial.clear();
        distribution.clear();
        return (return_initial,return_final,return_total);

    }//distributed call




}//end of mudule
