module queens_GPU_call_device_search{

	use queens_tree_exploration;
	use queens_node_module;
	use GPU_mlocale_utils;
	use CTypes;
	//use GPU_aux;
	use Math;
	//use CPtr;
	use Time;
  use GPUDiagnostics;

	config const CPUGPUVerbose: bool = false;

	proc queens_GPU_call_device_search(const num_gpus: c_int, const size: uint(16), const depthPreFixos: c_int,
		ref local_active_set: [] queens_node, const initial_num_prefixes: uint(64)): (uint(64), uint(64)) {

    startVerboseGPU();

		var vector_of_tree_size_h: [0..#initial_num_prefixes] c_ulonglong;
		var sols_h: [0..#initial_num_prefixes] c_ulonglong;

		//calculating the CPU load in terms of nodes
		var new_num_prefixes: uint(64) = initial_num_prefixes;
		var metrics: (uint(64),uint(64)) = (0:uint(64),0:uint(64));//

		//@question: should we use forall or coforall?

		for gpu_id in 0..#num_gpus:c_int do {
      var gpu_load: c_uint = GPU_mlocale_get_gpu_load(new_num_prefixes:c_uint, gpu_id:c_int, num_gpus);

      var starting_position = GPU_mlocale_get_starting_point(new_num_prefixes:c_uint,
        gpu_id:c_uint, num_gpus:c_uint, 0:c_uint);

      var sol_ptr : c_ptr(c_ulonglong) = c_ptrTo(sols_h) + starting_position;
      var tree_ptr : c_ptr(c_ulonglong) = c_ptrTo(vector_of_tree_size_h) + starting_position;
      var nodes_ptr : c_ptr(queens_node) = c_ptrTo(local_active_set) + starting_position;

      if(CPUGPUVerbose) then
        writeln("GPU id: ", gpu_id, " Starting position: ", starting_position, " gpu load: ", gpu_load);

        param _EMPTY_ = -1;

        on here.gpus[gpu_id] {

          var root_prefixes = local_active_set;
          var sols: [sols_h.domain] sols_h.eltType;
          var vector_of_tree_size: [vector_of_tree_size_h.domain] vector_of_tree_size_h.eltType;

          //writeln("starting loop");
          foreach idx in 0..#gpu_load {
            var flag = 0: uint(32);
            var bit_test = 0: uint(32);
            /*var board: [0..31] int(8);*/
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
              } else if (!(flag &  mask ) && GPU_queens_stillLegal(board, depth)){

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

          sols_h = sols;
          vector_of_tree_size_h = vector_of_tree_size;
        }
    }//end of gpu search

    stopVerboseGPU();

		var redTree = (+ reduce vector_of_tree_size_h):uint(64);
		var redSol  = (+ reduce sols_h):uint(64);

		return ((redSol,redTree)+metrics);
	}

  proc  GPU_queens_stillLegal(board, r) {
    var safe = true;
    const base = board[r];
    for (i, rev_i, offset) in zip(0..<r, 0..<r by -1, 1..r) {
      safe &= !((board[i] == base) || (board[rev_i] == base-offset ||
                                        board[rev_i] == base+offset));
    }
    return safe;
  }
}
