module queens_GPU_single_locale{

	//use GPU_aux;
	use queens_GPU_call_device_search;
	use queens_prefix_generation;
	use queens_node_module;
	use Time;

	use BlockDist;
	use VisualDebug;
	use CommDiagnostics;
	use DistributedIters;


	use CTypes;

	proc GPU_queens_call_search(const size: uint(16), const initial_depth: c_int){


		var initial_num_prefixes : uint(64);
		var initial_tree_size : uint(64) = 0;

		var initial, final: stopwatch;

		//search metrics
		var metrics: (uint(64),uint(64)) = (0:uint(64),0:uint(64));


		//starting the search
		initial.start();

		var maximum_number_prefixes: uint(64) = queens_get_number_prefixes(size,initial_depth);
		var local_active_set: [0..maximum_number_prefixes-1] queens_node;
		metrics+= queens_node_generate_initial_prefixes(size, initial_depth, local_active_set);

		initial.stop();


		initial_num_prefixes = metrics[0];
		initial_tree_size = metrics[1];

		metrics[0] = 0; //restarting for the parallel search_type
		metrics[1] = 0;

		////////////////////////////////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////////////

		writeln("\nSize: ", size, " Survivors: ", initial_num_prefixes);

		var num_gpus = here.gpus.size:c_int;
		writeln("Number of GPUs to use: ", num_gpus);
		if num_gpus == 0 then {
			writeln("#### No GPUs Found ####");
			halt();
		}


		final.start();

		metrics+=queens_GPU_call_device_search(num_gpus, size,
			initial_depth, local_active_set, initial_num_prefixes);

		final.stop();

		var final_tree_size = initial_tree_size+metrics[1];

		writeln("\n### End of the single-locale Multi-GPU search ###\n");
		writeln("Final tree size: ", final_tree_size);
		writeln("\tCPU tree size: ", initial_tree_size);
		writeln("\tGPU tree size: ", metrics[1]);
		writeln("Number of solutions: ", metrics[0]);
		writeln("Elapsed time: ", final.elapsed()+initial.elapsed(),"\n\n");

	}//single-locale-single-GPU search

}//
