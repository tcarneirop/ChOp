module bitset_mcore_search{

    use bitset_partial_search;
    use bitset_subproblem_module;
    use bitset_subproblem_explorer;
    use CTypes;
    use Time;
    use queens_aux;

    use DynamicIters;

    proc bitset_call_mcore_search(const board_size:int, const initial_depth:int, const chunk:int, const num_threads: int){

        var num_sols_search: uint(64) = 0;

        var initial_tree_size: uint(64) = 0;
        var final_tree_size: uint(64) = 0;
        var pool_size: uint(64) = 0;
        var metrics: (uint(64),uint(64)) =  (0,0);
       
        queens_print_initial_info(board_size:int(16), "dynamic",chunk,num_threads);

        //let's change this... very bad practice
        var subproblem_pool: [0..#999999] Bitqueens_subproblem;

        var timer: stopwatch;
        timer.start();

        metrics += queens_bitset_partial_search(board_size,initial_depth, subproblem_pool);
        pool_size = metrics[1];
        initial_tree_size = metrics[0];
        metrics = (0,0);
         
        writeln("\nInitial depth: ", initial_depth, "\nNumber of subproblems found (pool size): ", pool_size,"\n");
      
        forall subproblem in dynamic(0..#pool_size,chunk, num_threads) with (+reduce metrics) do{  
           metrics += queens_bitset_final_search(board_size, initial_depth, subproblem_pool[subproblem]);
        }

        timer.stop();
        metrics[0] <=> metrics[1];
        queens_print_serial_report(timer, board_size:uint(16), metrics,pool_size, initial_tree_size,metrics[0],initial_depth:int(32),"dynamic");

        writeln(metrics);
        writeln(timer.elapsed());
        
    }




    

}//module