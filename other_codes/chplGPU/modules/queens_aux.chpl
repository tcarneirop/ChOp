module queens_aux{

	use queens_serial;
	use Time; // Import the Time module to use Timer objects
    use SysCTypes;


	proc queens_serial_caller(const size: uint(16), const mode: string = "serial", const prepro: bool = false){

		use Time; // Import the Time module to use Timer objects
		var timer: Timer;
		var metrics: (uint(64),uint(64));
		timer.start(); // Start timer
		metrics = queens_serial_bitset(size);
		timer.stop(); // Start timer

		writeln("\nSerial Bitset Queens for size ", size);
		writeln("\tNumber of solutions: ", metrics[0]);
		writeln("\tTree size: ", metrics[1]);
		writeln("Elapsed time: ", timer.elapsed()*1000, " ms"); // Print elapsed time
		writeln("Rate: ", metrics[1]/timer.elapsed(), "n/s\n\n"); // Print elapsed time
		timer.clear(); // Clear timer for parallel loop

	}

	proc queens_print_initial_info(const size: uint(16), const scheduler: string, const lchunk: int = 1, const num_threads: int){

	writeln("N-Queens for size: ", size);
	writeln("\nCHPL Task layer: ", CHPL_TASKS,"\n\tNum created tasks: ",num_threads,"\n\tMax num tasks: ",here.maxTaskPar);
	writeln("\tScheduler: ", scheduler);
	writeln("\tLocal Chunk size: ", lchunk);

}//initial information

	proc queens_print_serial_report(timer: Timer, size: uint(16), metrics: (uint(64),uint(64)),
		initial_num_prefixes : uint(64), initial_tree_size: uint(64), parallel_tree_size: uint(64),
		const initial_depth: int(32),
        const scheduler: string){

        var number_of_solutions = metrics[0];
        var parallel_tree_size =  metrics[1];
        var final_tree_size = initial_tree_size + parallel_tree_size;
        var performance_metrics = (final_tree_size:real)/timer.elapsed();

        writeln("\n### Multicore N-Queens - ", scheduler ," ###\n\tProblem size (N): ", size,"\n\tCutoff depth: ",
         initial_depth,"\n\tInitial number of prefixes: ", initial_num_prefixes,
         "\n\n\tInitial tree size: ", initial_tree_size,
         "\n\tParallel tree size: ", parallel_tree_size,
         "\n\tFinal tree size: ", final_tree_size,
         "\n\n\tNumber of solutions found: ", number_of_solutions
         );

        writef("\n\nElapsed time: %.3dr", timer.elapsed());
        writef("\n\tPerformance: %.3dr (n/s)",  performance_metrics);
        writef("\n\tPerformance: %.3dr (solutions/s)\n\n",  (metrics[0]:real)/timer.elapsed());

	}//print serial report



}//module
