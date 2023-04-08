module queens_aux{

	use queens_serial;
	use Time; // Import the Time module to use Timer objects
    use statistics;
    use CTypes;
	use ChplConfig;

	proc queens_print_locales_information(){
        writeln("\nNumber of locales: ",numLocales,".");
        for loc in Locales do{
            on loc {
                writeln("\n\tLocale ", here.id, ", name: ", here.name,".");
            }
        }//end for
    }//print locales

	proc queens_serial_caller(const size: uint(16), const mode: string = "serial", const prepro: bool = false){

		use Time; 
		var timer: stopwatch;
		var metrics: (uint(64),uint(64));
		timer.start(); // Start timer

		if(mode=="first") then queens_first_serial_bitset( size: int, prepro);

		metrics = queens_serial_bitset(size);
		timer.stop(); // Start timer

		writeln("\nSerial Bitset Queens for size ", size);
		writeln("\tNumber of solutions: ", metrics[0]);
		writeln("\tTree size: ", metrics[1]);
		writeln("Elapsed time: ", timer.elapsed()*1000, " ms"); // Print elapsed time
		writeln("Rate: ", metrics[1]/timer.elapsed(), "n/s\n\n"); // Print elapsed time
		timer.clear(); // Clear timer for parallel loop

	}

	proc queens_print_serial_report(timer: stopwatch, size: uint(16), metrics: (uint(64),uint(64)),
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


	proc queens_print_initial_info(const size: uint(16), const scheduler: string, const lchunk: int = 1, const num_threads: int){

        writeln("N-Queens for size: ", size);
		writeln("\nCHPL Task layer: ", CHPL_TASKS,"\n\tNum created tasks: ",num_threads,"\n\tMax num tasks: ",here.maxTaskPar);
		writeln("\tScheduler: ", scheduler);
        writeln("\tLocal Chunk size: ", lchunk);

	}//initial information

    proc queens_print_mlocale_initial_info(const size: uint(16), initial_depth: c_int, second_depth: c_int,
        const scheduler: string, const lchunk: int, const mlchunk: int, const slchunk: int, const coordinated: bool,
        const num_threads: int, const mode: string, const pgas: bool){


        writeln("\n#### Distributed N-Queens #### \n\n\tSize: ", size);
        writeln("\tInitial depth: ",  initial_depth);
        writeln("\tSecond depth: ",   second_depth);

        writeln("\n#### PARAMETERS ####");
        writeln("\n\tCHPL Task layer: ", CHPL_TASKS,"\n\tNum tasks: ",num_threads,"\n\tMax tasks: ",here.maxTaskPar);


        writeln("\tCoordinated (centralized node): ", coordinated);
        writeln("\tDistributed active set (PGAS): ", pgas);
        writeln("\n\tDistributed scheduler: ", scheduler);
        writeln("\tMultilocale chunk size: ", mlchunk);
        writeln("\tTask chunk size (Mlocale): ", lchunk);
        writeln("\tSecond level chunk size (local): ", slchunk);
        writeln("\tSearch mode: ", mode);
        writeln("\n\n");

    }//initial information



	proc queens_mlocale_print_metrics(size: uint(16), ref metrics: (uint(64),uint(64)),
        ref initial: stopwatch, ref distribution: stopwatch, ref final: stopwatch, initial_tree_size: uint(64),
        maximum_num_prefixes: uint(64),initial_num_prefixes: uint(64), initial_depth: c_int,
        second_depth: c_int,  ref tree_each_locale: [] uint(64)){

        var performance_metrics: real = 0.0;
        var solutions_per_second: real = 0.0;
        var total_time: real = (final.elapsed()+initial.elapsed()+distribution.elapsed());
        var total_tree: uint(64) = (metrics[1]+initial_tree_size);


        performance_metrics = ((metrics[1]+initial_tree_size):real)/total_time;
        solutions_per_second = metrics[0]/total_time;

        writef("\n\tInitial depth: %u", initial_depth);
        writef("\n\tSecond depth: %u", second_depth);
        writef("\n\tMaximum possible prefixes: %u", maximum_num_prefixes);
        writef("\n\tInitial number of prefixes: %u", initial_num_prefixes);
        writef("\n\tPercentage of the maximum number: %.3dr\n",
            (initial_num_prefixes:real/maximum_num_prefixes:real)*100);

        writef("\n\tNumber of solutions found: %u", metrics[0]);
        writef("\n\tElapsed Initial Search: %.3dr", initial.elapsed());
        writef("\n\tElapsed PGAS Data Distribution: %.3dr", distribution.elapsed());
        writef("\n\tElapsed Final Search: %.3dr",     final.elapsed());
        writef("\n\tElapsed TOTAL: %.3dr\n",  final.elapsed()+initial.elapsed()+distribution.elapsed());
        writef("\n\tPGAS proportion: %.3dr\n", (distribution.elapsed())/(total_time)*100);


        writef("\n\tInitial Tree size: %u",initial_tree_size);
        writef("\n\tFinal Tree size: %u",  metrics[1]);
        writef("\n\tTOTAL Tree size: %u",  metrics[1]+initial_tree_size);
        writef("\n\n\tPerformance (nodes/s): %.3dr \n",  performance_metrics);
        writef("\n\n\tPerformance (solutions/s): %.3dr \n\n",  solutions_per_second);

        statistics_tree_statistics(tree_each_locale, total_tree);

    }//


}//module
