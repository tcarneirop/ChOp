module queens_call_mcore_search{

    use queens_tree_exploration;
    use queens_constants;
    use queens_node_module;
    use queens_prefix_generation;
    use DynamicIters;
    use Time; // Import the Time module to use Timer objects
    //config param methodStealing = Method.Whole; 
     config param methodStealing = Method.RoundRobin; 
    //config param methodStealing = Method.WholeTail;



    proc queens_node_call_search(size: uint(16), const initial_depth: int(32), 
        const scheduler: string, const chunk: int, const num_threads){

        var maximum_number_prefixes: uint(64) = queens_get_number_prefixes(size,initial_depth);
        var set_of_nodes: [0..maximum_number_prefixes-1] queens_node;
        var metrics: (uint(64),uint(64)) = (0:uint(64),0:uint(64));
        
        var initial_num_prefixes : uint(64) = 0;
        var initial_tree_size : uint(64) = 0;
        var number_of_solutions: uint(64) = 0;
        var final_tree_size: uint(64) = 0;
        var parallel_tree_size: uint(64) = 0;
        var performance_metrics: real = 0.0;
        var timer: stopwatch;

        queens_print_initial_info(size, scheduler,chunk,num_threads);
    
        timer.start(); // Start timer

        metrics += queens_node_generate_initial_prefixes(size,initial_depth, set_of_nodes );

        initial_num_prefixes = metrics[0];
        initial_tree_size = metrics[1];
        metrics[0] = 0; //restarting for the parallel search_type
        metrics[1] = 0;
        
        select scheduler{

            when "static" {
                forall idx in 0..#initial_num_prefixes with (+ reduce metrics) do {
                    metrics+=queens_subtree_explorer(size,initial_depth:int(32), set_of_nodes[idx:uint]);
                }
            }
            when "dynamic" {
                forall idx in dynamic(0..#initial_num_prefixes, chunk, num_threads) with (+ reduce metrics) do {
                    metrics+=queens_subtree_explorer(size,initial_depth:int(32), set_of_nodes[idx:uint]);
                }
            }
            when "guided" {
                forall idx in guided(0..#initial_num_prefixes,num_threads) with (+ reduce metrics) do {
                    metrics+=queens_subtree_explorer(size,initial_depth:int(32), set_of_nodes[idx:uint]);
                }
            }
            when "stealing" {
                forall idx in adaptive(0..#initial_num_prefixes,num_threads) with (+ reduce metrics) do {
                    metrics+=queens_subtree_explorer(size,initial_depth:int(32), set_of_nodes[idx:uint]);   
                }
            }
            otherwise{
                writeln("\n\n ###### error ######\n\n ###### error ######\n\n ###### error ###### ");
            }
        }//select


        timer.stop(); // Start timer

        queens_print_serial_report(timer, size, metrics,initial_num_prefixes, initial_tree_size,parallel_tree_size,
            initial_depth,scheduler);

        timer.clear();

   }//end of caller


}
