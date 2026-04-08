module bitset_mlocale_search{

    use bitset_partial_search;
    use bitset_subproblem_module;
    use bitset_subproblem_explorer;
    use queens_aux;
    use statistics;

    use CTypes;
    use Time;
    use BlockDist;
    use CTypes;
    use PrivateDist;
    use CyclicDist;
    use DistributedIters;	


    proc bitset_call_final_search(const board_size:int, const initial_depth:int, 
        const slchunk:int, const mlchunk:int, 
        const flag_coordinated: bool, const num_threads: int, 
        ref distributed_active_set: [] Bitqueens_subproblem, 
        const Space: domain(?), ref metrics: (uint(64),uint(64)),
        ref tree_each_locale: [] uint(64)
        ){

        forall idx in distributedDynamic(c=Space, chunkSize=slchunk,localeChunkSize=mlchunk,coordinated=flag_coordinated) with (+ reduce metrics) do {
            
            var m1 = queens_bitset_final_search(board_size, initial_depth, distributed_active_set[idx]);
            metrics+=m1;				
            on idx do tree_each_locale[here.id] += m1[0];
            
        }//for


    }


    proc bitset_call_mlocale_search(const board_size:int, const initial_depth:int, const slchunk:int, const mlchunk:int, 
        const flag_coordinated: bool, const num_threads: int, const pgas:bool = true){

        //writeln("################### Queens - distributed - naive - dynamic ###################");
        //writeln("################### Size: ", board_size," Initial depth: ", initial_depth," Second Level chunk: ", 
        //    slchunk," ML chunk:", mlchunk," Num Threads: ", num_threads," Coordinated? ", flag_coordinated, " ###################");
       

        var num_sols_search: uint(64) = 0;

        var initial_tree_size: uint(64) = 0;
        var final_tree_size: uint(64) = 0;
        var num_subproblems: uint(64) = 0;
        var metrics: (uint(64),uint(64)) =  (0,0);

        var tree_each_locale: [PrivateSpace] uint(64);

        queens_print_locales_information();
        //queens_print_mlocale_initial_info(size, initial_depth, second_depth, scheduler,lchunk, mlchunk, slchunk,
        //    coordinated,num_threads, mode,pgas);

        statistics_all_locales_init_explored_tree(tree_each_locale);
       

        //let's change this

        var total, initial, final, distribution: stopwatch;
        
        total.start();

        var subproblem_pool: [0..#999999] Bitqueens_subproblem;

        metrics += queens_bitset_partial_search(board_size,initial_depth, subproblem_pool);
        initial_tree_size = metrics[0];
        num_subproblems = metrics[1];
        metrics[1] = 0;

        var rangeDynamic: range = 0..#num_subproblems;

        //Distributer or centralized active set?
        const Space = {0..#num_subproblems}; //for distributing
        const D: domain(1) dmapped new blockDist(boundingBox=Space) = Space; //1d block DISTRIBUTED
        var pgas_active_set: [D] Bitqueens_subproblem; //1d block DISTRIBUTED
        //var centralized_active_set: [Space] bitset_subproblem; //on node 0

        writeln("####  initialization of the Active Set  ####");
        //forall i in Space do centralized_active_set[i] = subproblem_pool[i];

        if(pgas) then {
            writeln("#####  PGAS-based active set #####");
            pgas_active_set =  subproblem_pool[Space];
            bitset_call_final_search( board_size, initial_depth, slchunk,mlchunk, 
                flag_coordinated,  num_threads, pgas_active_set, 
                Space,metrics,tree_each_locale);

        }
        else{
            bitset_call_final_search( board_size, initial_depth, slchunk,mlchunk, 
                flag_coordinated, num_threads, subproblem_pool, 
                Space,metrics,tree_each_locale);
            writeln("#####  Centralized active set #####");
        }

        

        total.stop();
        metrics[1]*=2;
        writeln("\n\n###################################");
        writeln("Tree size: ",metrics[0]);
        writeln("Number of solutions found: ", metrics[1]);
        writeln("Execution time: ",total.elapsed(),"\n");
        
        metrics[0]<=>metrics[1];
        metrics[0] = metrics[0]/2;

        queens_mlocale_print_metrics(board_size:uint(16), metrics, initial, distribution,total,
            initial_tree_size, 99999999, num_subproblems , initial_depth:int(32), 0, tree_each_locale);

        
    }




    

}//module