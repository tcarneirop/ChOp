module queens_CHPL_call_device_search{

	use queens_tree_exploration;
	use queens_node_module;
	use GPU_mlocale_utils;
	use CTypes;
	use Math;
	use Time;
	use GpuDiagnostics;
	use GPU;

	proc queens_CHPL_call_device_search(const num_gpus: c_int, const size: uint(16), const depthPreFixos: c_int,
			ref local_active_set: [] queens_node, const initial_num_prefixes: uint(64)): (uint(64), uint(64)) {

		//startVerboseGpu();

		//writeln("initial_num_prefixes:", initial_num_prefixes);
		
		//calculating the CPU load in terms of nodes
		var new_num_prefixes: uint(64) = initial_num_prefixes;
		var metrics: (uint(64),uint(64)) = (0:uint(64),0:uint(64));
		
		var reduce_tree_size: [0..#num_gpus] c_ulonglong = 0;
		var reduce_num_sols: [0..#num_gpus] c_ulonglong = 0;

		var  tree_size_h: [0..#num_gpus] c_ulonglong = 0;
		var  num_sols_h: [0..#num_gpus] c_ulonglong = 0;

		coforall gpu_id in 0..#num_gpus:c_int do {
			
			var gpu_load: c_uint = GPU_mlocale_get_gpu_load(new_num_prefixes:c_uint, gpu_id:c_int, num_gpus);

			var starting_position: c_uint = GPU_mlocale_get_starting_point(new_num_prefixes:c_uint,
					gpu_id:c_uint, num_gpus:c_uint, 0:c_uint);
			
			var my_load = starting_position..#(gpu_load); /// !!! HERE !!! -- I dont know if this range is correct... but it is working... 
			

			//writeln("Total num prefixes: ",new_num_prefixes," GPU id: ", gpu_id,"starting_position: ", starting_position, " My load:   ", my_load , " GPU load: ", gpu_load,  "   " ,gpu_id..#gpu_id);
	  
			param _EMPTY_ = -1;

			on here.gpus[gpu_id] {
				
				var root_prefixes = local_active_set[my_load]; //!!!! HERE !!!!! [my_load] to the other ones
				
				var sols: [my_load] c_ulong; //!!!! HERE !!!!!
				var vector_of_tree_size: [my_load] c_ulong; //!!!! HERE !!!!!


				//writeln("starting loop");
				foreach idx in my_load{ 

					//setBlockSize(512);
				
					//assertOnGpu();

					var flag = 0: uint(32);
		
					var board: c_array(int(8), 32);

					var depth: int(32);

					var N_l = size;
					var qtd_solucoes_thread = 0: uint(64);
					var depthGlobal = depthPreFixos;
					var tree_size = 0: uint(64);

					for i in 0..<N_l do  // what happens if I use promotion here?
						board[i] = _EMPTY_;

					flag = root_prefixes[idx].control;

					for i in 0..<depthGlobal do
						board[i] = root_prefixes[idx].board[i];

					depth=depthGlobal;

					do{
						board[depth] += 1;
						const mask = 1:int(32)<<board[depth];

						if(board[depth] == N_l){
							board[depth] = _EMPTY_;
							//if(block_ub > upper)   block_ub = upper;
							depth -= 1;
							flag &= ~(1:int(32)<<board[depth]);
						} else if (!(flag & mask ) && CHPL_queens_stillLegal(board, depth)){

							tree_size += 1;
							flag |= mask;

							depth += 1;

							if (depth == N_l) { //sol
								qtd_solucoes_thread += 1;

								depth -= 1;
								flag &= ~mask;
							}
						}

					}while(depth >= depthGlobal); //FIM DO DFS_BNB

					/*writeln("Sols: ", qtd_solucoes_thread);*/
					sols[idx] = qtd_solucoes_thread;
					vector_of_tree_size[idx] = tree_size;
					
				}

				reduce_tree_size[gpu_id] = gpuSumReduce(vector_of_tree_size); //!!!! HERE !!!!!
				reduce_num_sols[gpu_id] =  gpuSumReduce(sols);	//!!!! HERE !!!!!
				
			}//for idx in myload
			
		}//end of gpu search

		//stopVerboseGpu();

		var redTree = (+ reduce reduce_tree_size):uint(64); //!!!! HERE !!!!!
		var redSol  = (+ reduce reduce_num_sols):uint(64); //!!!! HERE !!!!!

		return ((redSol,redTree)+metrics);
	}

	inline proc CHPL_queens_stillLegal(board, r) {
		var safe = true;
		const base = board[r];
		for (i, rev_i, offset) in zip(0..<r, 0..<r by -1, 1..r) {
			safe &= !((board[i] == base) | ( (board[rev_i] == base-offset) |
						(board[rev_i] == base+offset)));
		}
		return safe;
	}
}
