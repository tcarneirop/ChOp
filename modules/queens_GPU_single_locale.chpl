module queens_GPU_single_locale{

	use GPU_aux;
	use queens_GPU_call_device_search;
	use queens_prefix_generation;
	use queens_node_module;
	use queens_aux;
	use Time;
	use queens_CHPL_call_device_search;

	use BlockDist;
	use VisualDebug;
	use CommDiagnostics;
	use DistributedIters;


	use CTypes;

	proc GPU_queens_call_search(const num_gpus: c_int, const size: uint(16), const initial_depth: c_int, 
		const CPUP: real,const lchunk:int, const language: string){


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

		writeln("\nSize: ", size, " Initial Pool Size at depth ", initial_depth," : ",initial_num_prefixes);

		//var num_gpus = GPU_device_count();

		writeln("Number of GPUs to use: ", num_gpus);
		writeln("Implementation: ", language);

		writeln("Percentage of the active set on the CPU: ", CPUP*100.0);

		final.start();

		select language{
			when "chpl"{
				writeln("Chapel-GPU");
				metrics+= queens_CHPL_call_device_search(num_gpus, size, initial_depth, local_active_set,
					initial_num_prefixes);
			}
			//for both amd and CUDA
			otherwise{
				//writeln(language);
				metrics+=queens_GPU_call_device_search(num_gpus, size,
					initial_depth, local_active_set, initial_num_prefixes, CPUP,lchunk);
			} 
		}

		final.stop();

		var final_tree_size = initial_tree_size+metrics[1];

		writeln("\n### End of the single-locale Multi-GPU search ###\n");
		writeln("Final tree size: ", final_tree_size);
		writeln("\tCPU tree size: ", initial_tree_size);
		writeln("\tGPU tree size: ", metrics[1]);

    	writeln("Number of solutions: ", metrics[0]*2);
		writeln("Elapsed time: ", final.elapsed()+initial.elapsed(),"\n\n");
		writeln("\tInitial search el. time: ", initial.elapsed());
		writeln("\tFinal search el. time: ",  final.elapsed());

	}//single-locale-single-GPU search

}//
