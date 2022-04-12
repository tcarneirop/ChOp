
module fsp_aux{

	use Time;
    use CTypes;
	use ChplConfig;




    var optimal: [1..120] int = [1278,1359,1081,1293,1235,1195,1234,1206,1230,1108,
        1582,1659,1496,1377,1419,1397,1484,1538,1593,1591,
        2297,2099,2326,2223,2291,2226,2273,2200,2237,2178,
        2724,2834,2621,2751,2863,2829,2725,2683,2552,2782,
        2991,2867,2839,3063,2976,3006,3093,3037,2897,3065,
        3850,3704,3640,3723,3611,3681,3704,3691,3743,3756,
        5493,5268,5175,5014,5250,5135,5246,5094,5448,5322,
        5770,5349,5676,5781,5467,5303,5595,5617,5871,5845,
        6202,6183,6271,6269,6314,6364,6268,6401,6275,6434,
        10862,10480,10922,10889,10524,10329,10854,10730,10438,10675,
        11195,11203,11281,11275,11259,11176,11360,11334,11192,11288,
        26059,26520,26371,26456,26334,26477,26389,26560,26005,26457,
        ];



    proc fsp_is_aset_empty(initial_num_prefixes:  uint(64), initial_tree_size : uint(64) = 0 ){

        if(initial_num_prefixes == 0:uint(64)){
            writeln(" \n\n ### empty active set ### ");
            writeln("\t Resulting tree size: ", initial_tree_size);
            exit(1);
        }

    }

    proc fsp_get_number_prefixes(const jobs: c_int, const initial_depth: c_int): uint(64){

       var number: uint(64) = jobs:uint(64);

       if initial_depth == 1 then
           return number;

       for d in 2..initial_depth do
           number *= (((jobs-d)+1):uint(64));

       return number;

    }//get maximum number of prefixes at depth d


    proc fsp_get_upper_bound(upper_bound: c_int, instance: c_short): c_int{

        if upper_bound == 0 then
            return optimal[instance:int]:c_int;
        else
            return upper_bound;
    }


    proc fsp_print_serial_report(timer: Timer, machines: c_int, jobs: c_int,
        metrics: (uint(64),uint(64),c_int), upper_bound: c_int){

        var final_tree_size: uint(64) = 0;
        var performance_metrics: real = 0.0;
        performance_metrics = (metrics[1]:real)/timer.elapsed();

        writeln("Machines: ", machines);
        writeln("Jobs: ", jobs);
        writef("\n\nElapsed time: %.3dr", timer.elapsed());
        writef("\n\tNumber of solutions found: %u", metrics[0]);
        writef("\n\tInitial solution: %i", upper_bound);
        writef("\n\tBest solution: %i", metrics[2]:uint(32));
        writef("\n\tTree size: %u", metrics[0]);
        writef("\n\tPerformance: %.3dr (n/s)\n\n\n",  performance_metrics);

    }//print serial report


    proc fsp_print_initial_info(const scheduler: string, const chunk: int = 1, const num_threads: int){

        writeln("\nCHPL Task layer: ", CHPL_TASKS,"\n\tNum created tasks: ",num_threads,"\n\tMax num tasks: ",here.maxTaskPar);
        writeln("\tScheduler: ", scheduler);
        if(scheduler == "dynamic"){
            writeln("\tChunk size: ", chunk);
        }

    }//initial information

    proc fsp_print_mcore_initial_info(initial_depth: c_int, upper_bound: c_int,
        const scheduler: string, const lchunk: int,const num_threads: int, const instance: c_short){

        writeln("\n#### Multicore BnB #### \n\n\tInstance: ", instance, "\n\tUpper bound: ", upper_bound);
        writeln("\tInitial depth: ",  initial_depth);
        writeln("\n#### PARAMETERS ####");
        writeln("\tCHPL Task layer: ", CHPL_TASKS,"\n\tNum tasks: ",num_threads,"\n\tMax tasks: ",here.maxTaskPar);
        writeln("\tScheduler: ", scheduler);
        writeln("\tLocal chunk size: ", lchunk);

    }//initial information


    proc fsp_new_print_initial_info(initial_depth: c_int, second_depth: c_int, upper_bound: c_int,
        const scheduler: string,
        const lchunk: int, const mlchunk: int, const slchunk: int,  const coordinated: bool,
        const num_threads: int,const atype: string, const instance: c_short,
        const mode: string, const pgas: bool){


        writeln("\n#### Distributed BnB #### \n\n\tInstance: ", instance, "\n\tUpper bound: ", upper_bound,"\n");
        writeln("\tInitial depth: ",  initial_depth);
        writeln("\tSecond depth: ",   second_depth);

        writeln("\n#### PARAMETERS ####");
        writeln("\n\tCHPL Task layer: ", CHPL_TASKS,"\n\tNum tasks: ",num_threads,"\n\tMax tasks: ",here.maxTaskPar);

        writeln("\n\tDistributed scheduler: ", scheduler);
        writeln("\tDistributed active set (PGAS): ", pgas);
        writeln("\tMultilocale chunk size: ", mlchunk);
        writeln("\tTask chunk size (Mlocale): ", lchunk);
        writeln("\tSecond level chunk size (local): ", slchunk);
        writeln("\tCoordinated (centralized node): ", coordinated);
        writeln("\tSearch mode: ", mode);
        writeln("\tAtomic type: ", atype);
        writeln("\n");

    }//initial information


	proc fsp_print_metrics( machines: c_int, jobs: c_int, ref metrics: (uint(64),uint(64)),
        ref initial: Timer, ref final: Timer, initial_tree_size: uint(64), maximum_num_prefixes: uint(64),initial_num_prefixes: uint(64),
        initial_ub:c_int, ref final_ub: atomic c_int ){

        var performance_metrics: real = 0.0;

        performance_metrics = ((metrics[1]+initial_tree_size):real)/(final.elapsed() + initial.elapsed());

        writeln("### Metrics ###");
        writef("\n\tMaximum possible prefixes: %u", maximum_num_prefixes);
        writef("\n\tInitial number of prefixes: %u", initial_num_prefixes);
        writef("\n\tPercentage of the maximum number: %.3dr\n",
        	(initial_num_prefixes:real/maximum_num_prefixes:real)*100);

        writef("\n\tNumber of solutions found: %u", metrics[0]);
        writef("\n\tInitial solution: %i", initial_ub);
        writef("\n\tOptimal solution: %i\n", final_ub.read());

        writef("\n\tElapsed Initial: %.3dr", initial.elapsed());
        writef("\n\tElapsed Final: %.3dr",   final.elapsed());
        writef("\n\tElapsed TOTAL: %.3dr\n",   final.elapsed()+initial.elapsed());

        writef("\n\tTree size: %u",  metrics[1]+initial_tree_size);
        writef("\n\tPerformance: %.3dr (n/s)\n\n",  performance_metrics);

    }//

}//module
