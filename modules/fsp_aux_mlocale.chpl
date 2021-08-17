module fsp_aux_mlocale{


	use Time;
    use SysCTypes;
    use fsp_node_module;
    use statistics;
    use List;

    proc print_locales_information(){
        writeln("Number of locales: ",numLocales,".");
        for loc in Locales do{
            on loc do{
                writeln("\tLocale ", here.id, ", name: ", here.name,".");
            }
        }//end for
    }//print locales

    proc fsp_all_locales_init_ub(const upper_bound: c_int, ref global_ub: atomic c_int, 
        ref array_upper_bound: [] atomic c_int){

        writeln("# Starting the initial upper bound on all locales: ", upper_bound, ".");
        global_ub.write(upper_bound);

        forall a in array_upper_bound do
            a.write(upper_bound);
    }/////


	proc fsp_all_locales_print_minTempsArr(machines: c_int){
  		
        for loc in Locales do{
            on loc do{//but locale one -- let's put it
            	writeln("MinTempsArr on Locale #", here.id);
                for i in 0..machines-1 do
                    writeln(minTempsArr[i]);
                writeln("\n\n\n");
            }
        }
	}//print mintemparr


	proc fsp_all_locales_print_minTempsDep(machines: c_int){
  		
        for loc in Locales do{
            on loc do{//but locale one -- let's put it
            	writeln("MinTempsDep on Locale #", here.id);
                for i in 0..machines-1 do
                    writeln(minTempsDep[i]);
                writeln("\n\n\n");
            }
        }
	}//print mintempsdep


    proc fsp_mlocale_print_metrics(instance: c_short, machines: c_int, jobs: c_int, ref metrics: (uint(64),uint(64)), 
        ref initial: Timer, ref inialization: Timer, ref distribution: Timer, ref final: Timer, initial_tree_size: uint(64), 
        maximum_num_prefixes: uint(64),initial_num_prefixes: uint(64), 
        initial_ub:c_int, final_ub: c_int, initial_depth: c_int, second_depth: c_int,
        ref tree_each_locale: [] uint(64), ml_chunk: int){
        
        var performance_metrics: real = 0.0;
        var total_tree: uint(64) = (metrics[1]+initial_tree_size);


        performance_metrics = ((metrics[1]+initial_tree_size):real)/(final.elapsed()+initial.elapsed()+inialization.elapsed()+distribution.elapsed());
        
        writeln("\nInstace: ta",instance);
        writeln("Machines: ", machines);
        writeln("Jobs: ", jobs);

        writef("\n\tInitial depth: %u", initial_depth);
        writef("\n\tSecond depth: %u", second_depth);
        writef("\n\tMaximum possible prefixes: %u", maximum_num_prefixes);
        writef("\n\tInitial number of prefixes: %u", initial_num_prefixes);
        writef("\n\tPercentage of the maximum number: %.3dr\n", 
            (initial_num_prefixes:real/maximum_num_prefixes:real)*100);

        writef("\n\tNumber of solutions found: %u", metrics[0]);
        writef("\n\tInitial solution: %i", initial_ub);
        writef("\n\tOptimal solution: %i\n", final_ub);
        
        writef("\n\tElapsed Initial Search: %.3dr", initial.elapsed());
        writef("\n\tElapsed PGAS Data Initialization: %.3dr", inialization.elapsed());
        writef("\n\tElapsed PGAS Data Distribution: %.3dr", distribution.elapsed());
        writef("\n\tElapsed Final Search: %.3dr",     final.elapsed());
        writef("\n\tElapsed TOTAL: %.3dr\n",  final.elapsed()+initial.elapsed()+inialization.elapsed()+distribution.elapsed());
        writef("\n\tPGAS proportion: %.3dr\n", (inialization.elapsed()+distribution.elapsed())/(final.elapsed()+initial.elapsed()+inialization.elapsed()+distribution.elapsed())*100);

        
        writef("\n\tInitial Tree size: %u",initial_tree_size);
        writef("\n\tFinal Tree size: %u",  metrics[1]);
        writef("\n\tTOTAL Tree size: %u",  metrics[1]+initial_tree_size);
        writef("\n\n\tPerformance: %.3dr (n/s)\n\n",  performance_metrics);


       // writeln("\nChunk weight: ");
       // writeln("\rChunk size: ", ml_chunk);
       // for i in 0..#numLocales do{
       // 	writeln(chunk_weight[i]);
       //     writeln("Locale ", i, ": ");
       //     //writeln(chunk_weight[i]);
       //     writeln("\t Biggest subtree: ", max reduce chunk_weight[i]);
       //     //writeln("\t Smalles subtree: ", min reduce chunk_weight[i]);
       //     writeln("\t Number of chunks: ", chunk_weight[i].size/ml_chunk + (if (chunk_weight[i].size%ml_chunk) then 1 else 0));
        //}

        statistics_tree_statistics(tree_each_locale, total_tree);
        
    }//


    proc fsp_call_initial_print_metrics( instance: c_short, machines: c_int, jobs: c_int, 
        ref initial: Timer, ref inialization: Timer, ref distribution: Timer, initial_tree_size: uint(64), 
        maximum_num_prefixes: uint(64),initial_num_prefixes: uint(64), initial_ub:c_int){
        
        var performance_metrics: real = 0.0;

        performance_metrics = ((initial_tree_size):real)/(initial.elapsed()+inialization.elapsed()+distribution.elapsed());
        
        writeln("\nInstace: ta", instance);
        writeln("Machines: ", machines);
        writeln("Jobs: ", jobs);

        writef("\n\tMaximum possible prefixes: %u", maximum_num_prefixes);
        writef("\n\tInitial number of prefixes: %u", initial_num_prefixes);
        writef("\n\tPercentage of the maximum number: %.3dr\n", 
            (initial_num_prefixes:real/maximum_num_prefixes:real)*100);

        writef("\n\tInitial solution: %i", initial_ub);
        
        writef("\n\tElapsed Initial: %.3dr", initial.elapsed());
        writef("\n\tElapsed PGAS Data Initialization: %.3dr", inialization.elapsed());
        writef("\n\tElapsed PGAS Data Distribution: %.3dr", distribution.elapsed());
        writef("\n\tElapsed TOTAL: %.3dr\n", initial.elapsed()+inialization.elapsed()+distribution.elapsed());

        writef("\n\tTree size: %u",  initial_tree_size);
        writef("\n\tPerformance: %.3dr (n/s)\n\n",  performance_metrics);
        
    }//





}//module
	