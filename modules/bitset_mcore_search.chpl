module bitset_mcore_search{

    use bitset_partial_search;
    use bitset_subproblem_module;
    use bitset_subproblem_explorer;
    use CTypes;
    use Time;

    use DynamicIters;

    proc bitset_call_mcore_search(const board_size:int, const initial_depth:int, const chunk:int, const num_threads: int){

        var num_sols_search: uint(64) = 0;

        var initial_tree_size: uint(64) = 0;
        var final_tree_size: uint(64) = 0;
        var pool_size: uint(64) = 0;
        var metrics: (uint(64),uint(64)) =  (0,0);
       

        //let's change this
        var subproblem_pool: [0..#99999] Bitqueens_subproblem;

        var timer: stopwatch;
        timer.start();

        metrics += queens_bitset_partial_search(board_size,initial_depth, subproblem_pool);
        pool_size = metrics[1];
        metrics[1] = 0;


        forall subproblem in dynamic(0..#pool_size,chunk, num_threads) with (+reduce metrics) do{  
        //forall subproblem in 0..#pool_size with (+reduce metrics) do{
           metrics += queens_bitset_final_search(board_size, initial_depth, subproblem_pool[subproblem]);
        }

        timer.stop();

        writeln(metrics);
        writeln(timer.elapsed());
        
    }




    

}//module