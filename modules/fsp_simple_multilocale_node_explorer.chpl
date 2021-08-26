module fsp_simple_multilocale_node_explorer{
	use CPtr;	
	use fsp_node_module;
	use fsp_simple_chpl_c_headers;
	use fsp_constants;
	use SysCTypes;
	use concurrency;	


	proc fsp_simple_mlocale_array_node_explorer(const machines: c_int, const jobs: c_int, 
		const initial_depth: c_int, ref node: fsp_node, ref global_upper_bound: atomic c_int, 
		ref set_of_atomics: [] atomic c_int): (uint(64),uint(64)){

      	var depth: c_int; //needs to be int because -1 is the break condition

		//one for each thread
        var front: [0.._MAX_MACHINES_] c_int;//private - search
		var back: [0.._MAX_MACHINES_] c_int;
		var remain: [0.._MAX_MACHINES_] c_int;

      	//state of the search 	//SEARCH initialization		
      	var control: [0..(jobs)-1] bool = [i in 0..(jobs)-1] node.control[i];
      	var scheduled: [0..(jobs)-1] c_int = [i in 0..(jobs)-1] node.scheduled[i];
      	var position: [0..(jobs)-1] c_int = [i in 0..(jobs)-1] node.position[i];
      	var permutation: [0..(jobs)-1] c_int = [i in 0..(jobs)-1] node.permutation[i];
	
		//aux 
		var incumbent: c_int;
    	var lowerbound: c_int = 0;
    	var p1: c_int;

	    //CONTROL
		var num_sols: uint(64) = 0;
		var tree_size: uint(64) = 0;
		var metrics: (uint(64),uint(64));

		depth = initial_depth;


		while(true){//Search

			scheduled[depth] = scheduled[depth]+1;

			if scheduled[depth] == jobs then
				scheduled[depth] = _FSP_EMPTY_;
			else{
				if (!control[scheduled[depth]]) {

					p1 = permutation[depth];
                    swap(permutation[depth],permutation[position[scheduled[depth]]]);
                    swap(position[scheduled[depth]],position[p1]);

                    incumbent = set_of_atomics[here.id].read();

                    lowerbound = simple_bornes_calculer(c_ptrTo(permutation), depth, jobs,
                         machines, jobs, c_ptrTo(remain), c_ptrTo(front), c_ptrTo(back), 
                         minTempsArr_s, minTempsDep_s, c_temps);

                    if(lowerbound<incumbent){

						control[scheduled[depth]] = true;
						depth +=1;
						tree_size+=1;

						if (depth == jobs){
							num_sols+=1;
							incumbent = concurrency_mlocale_minExchange(lowerbound, set_of_atomics[here.id], global_upper_bound);
							writeln("New solution: ", incumbent );
							//incumbent = mlocale_concurrency_minExchange(lowerbound, global_upper_bound, set_of_atomics[here.id]);
						}
						else continue;
					}//
					else continue;
				}//if valid
				else continue;
			}//else

			depth -= 1;
			control[scheduled[depth]] = false;

			if (depth < initial_depth) then
				break;
		}//search

		metrics[0] = num_sols;
	    metrics[1] = tree_size;

		return metrics;

	}//node explorer


	proc fsp_simple_mlocale_global_atomic_node_explorer(const machines: c_int, const jobs: c_int, 
		const initial_depth: c_int, ref node: fsp_node, ref global_upper_bound: atomic c_int ): (uint(64),uint(64)){

      	var depth: c_int; //needs to be int because -1 is the break condition

		//one for each thread
		var front: [0.._MAX_MACHINES_] c_int;//private - search
		var back: [0.._MAX_MACHINES_] c_int;
		var remain: [0.._MAX_MACHINES_] c_int;
      	//state of the search 	//SEARCH initialization		
      	var control: [0..(jobs)-1] bool = [i in 0..(jobs)-1] node.control[i];
      	var scheduled: [0..(jobs)-1] c_int = [i in 0..(jobs)-1] node.scheduled[i];
      	var position: [0..(jobs)-1] c_int = [i in 0..(jobs)-1] node.position[i];
      	var permutation: [0..(jobs)-1] c_int = [i in 0..(jobs)-1] node.permutation[i];
	
		//aux 
		// var incumbent: c_int = global_upper_bound.read();
		var incumbent: c_int;
    	var lowerbound: c_int = 0;
    	var p1: c_int;

	    //CONTROL
		var num_sols: uint(64) = 0;
		var tree_size: uint(64) = 0;
		var metrics: (uint(64),uint(64));

		//SEARCH initialization		
		depth = initial_depth;

		while(true){//Search

			scheduled[depth] = scheduled[depth]+1;

			if scheduled[depth] == jobs then
				scheduled[depth] = _FSP_EMPTY_;
			else{
				if (!control[scheduled[depth]]) {

					incumbent = global_upper_bound.read();

					p1 = permutation[depth];
                    swap(permutation[depth],permutation[position[scheduled[depth]]]);
                    swap(position[scheduled[depth]],position[p1]);

                    lowerbound = simple_bornes_calculer(c_ptrTo(permutation), depth, jobs,
                         machines, jobs, c_ptrTo(remain), c_ptrTo(front), c_ptrTo(back), 
                         minTempsArr_s, minTempsDep_s, c_temps);

                    if(lowerbound<incumbent){

						control[scheduled[depth]] = true;
						depth +=1;
						tree_size+=1;

						if (depth == jobs){
							num_sols+=1;
							incumbent = new_concurrency_minExchange(lowerbound, global_upper_bound);
							writeln("New solution: ", incumbent );
						}
						else continue;
					}//
					else continue;
				}//if valid
				else continue;
			}//else

			depth -= 1;
			control[scheduled[depth]] = false;

			if (depth < initial_depth) then
				break;
		}//search

		metrics[0] = num_sols;
	    metrics[1] = tree_size;

		return metrics;

	}//node explorer


	proc fsp_simple_mlocale_node_explorer(const machines: c_int, const jobs: c_int, 
		const initial_depth: c_int, ref node: fsp_node, global_upper_bound:c_int ): (uint(64),uint(64)){

      	var depth: c_int; //needs to be int because -1 is the break condition

		//one for each thread
	 	var front: [0.._MAX_MACHINES_] c_int;//private - search
		var back: [0.._MAX_MACHINES_] c_int;
		var remain: [0.._MAX_MACHINES_] c_int;

      	//state of the search 	//SEARCH initialization		
      	var control: [0..(jobs)-1] bool = [i in 0..(jobs)-1] node.control[i];
      	var scheduled: [0..(jobs)-1] c_int = [i in 0..(jobs)-1] node.scheduled[i];
      	var position: [0..(jobs)-1] c_int = [i in 0..(jobs)-1] node.position[i];
      	var permutation: [0..(jobs)-1] c_int = [i in 0..(jobs)-1] node.permutation[i];

		//aux 
		// var incumbent: c_int = global_upper_bound.read();
		var incumbent: c_int = global_upper_bound;
    	var lowerbound: c_int = 0;
    	var p1: c_int;

	    //CONTROL
		var num_sols: uint(64) = 0;
		var tree_size: uint(64) = 0;
		var metrics: (uint(64),uint(64));

		//SEARCH initialization		
		depth = initial_depth;

		while(true){//Search

			scheduled[depth] = scheduled[depth]+1;

			if scheduled[depth] == jobs then
				scheduled[depth] = _FSP_EMPTY_;
			else{
				if (!control[scheduled[depth]]) {

					p1 = permutation[depth];
                    swap(permutation[depth],permutation[position[scheduled[depth]]]);
                    swap(position[scheduled[depth]],position[p1]);

                    lowerbound = simple_bornes_calculer(c_ptrTo(permutation), depth, jobs,
                         machines, jobs, c_ptrTo(remain), c_ptrTo(front), c_ptrTo(back), 
                         minTempsArr_s, minTempsDep_s, c_temps);

                    if(lowerbound<incumbent){

						control[scheduled[depth]] = true;
						depth +=1;
						tree_size+=1;

						if (depth == jobs){
							num_sols+=1;
						}
						else continue;
					}//
					else continue;
				}//if valid
				else continue;
			}//else

			depth -= 1;
			control[scheduled[depth]] = false;

			if (depth < initial_depth) then
				break;
		}//search

		metrics[0] = num_sols;
	    metrics[1] = tree_size;

		return metrics;

	}//node explorer


}