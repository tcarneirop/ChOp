
module queens_CHPL_call_device_search{

	use queens_tree_exploration;
	use queens_node_module;
	use GPU_mlocale_utils;
	use CTypes;
	use Math;
	use Time;
	use GpuDiagnostics;
	use GPU;
	use DynamicIters;
	config param CHPL_CPUGPUVerbose: bool = false;

	proc queens_CHPL_call_device_search(const num_gpus: c_int, const size: uint(16), const depthPreFixos: c_int,
			ref local_active_set: [] queens_node, const initial_num_prefixes: uint(64),
			const CPUP: real, const chunk: int
			): (uint(64), uint(64)) {

		//startVerboseGpu();

		//calculating the CPU load in terms of nodes
		var cpu_load: c_uint = (CPUP * initial_num_prefixes):c_uint;
		var new_num_prefixes: uint(64) = initial_num_prefixes - cpu_load:uint(64);
		var metrics: (uint(64),uint(64)) = (0:uint(64),0:uint(64));
		
		var reduce_tree_size: [0..#num_gpus] c_ulonglong = 0;
		var reduce_num_sols: [0..#num_gpus] c_ulonglong = 0;

		var  tree_size_h: [0..#num_gpus] c_ulonglong = 0;
		var  num_sols_h: [0..#num_gpus] c_ulonglong = 0;

		cobegin with (ref metrics){

			{/////
				if(CHPL_CPUGPUVerbose){//use this variable to see the debug messages
					writeln("CPUP: ", CPUP);
					writeln("Going on CPU");
				}

				forall idx in dynamic(0..(cpu_load:int), chunk,here.maxTaskPar) with (+ reduce metrics ) do {
					metrics +=  queens_subtree_explorer(size,depthPreFixos,local_active_set[idx:uint]);
				}

				if(CHPL_CPUGPUVerbose){
					writeln("End of the CPU search.");
				}

			}////


			coforall gpu_id in 0..#num_gpus:c_int do {
				
				var gpu_load: c_uint = GPU_mlocale_get_gpu_load(new_num_prefixes:c_uint, gpu_id:c_int, num_gpus);

				var starting_position: c_uint = GPU_mlocale_get_starting_point(new_num_prefixes:c_uint, gpu_id:c_uint, num_gpus:c_uint, 0:c_uint);
				
				var my_load = starting_position..#(gpu_load); 
				
				var new_gpu_id: c_uint;

				if Locales.size == 1 then new_gpu_id = gpu_id:c_int; else new_gpu_id = (here.id:c_int)%(here.gpus.size:c_int);

				param _EMPTY_ = -1;

				on here.gpus[new_gpu_id] {
					
					var root_prefixes = local_active_set[my_load]; 
					
					var sols: [my_load] c_ulong; 
					var vector_of_tree_size: [my_load] c_ulong; 

					//writeln("starting loop");
					foreach idx in my_load{ 

						//setBlockSize(256);
					
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

					reduce_tree_size[gpu_id] = gpuSumReduce(vector_of_tree_size); 
					reduce_num_sols[gpu_id] =  gpuSumReduce(sols);
					
				}//for idx in myload
				
			}//end of gpu search

		}



		//stopVerboseGpu();

		var redTree = (+ reduce reduce_tree_size):uint(64); 
		var redSol  = (+ reduce reduce_num_sols):uint(64); 

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
