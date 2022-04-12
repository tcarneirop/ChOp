module fsp_simple_node_explorer{

	use fsp_node_module;
	use fsp_simple_chpl_c_headers;
	use fsp_constants;
	use CTypes;
	use concurrency;
	//use CPtr;

	proc fsp_simple_node_explorer(const machines: c_int, const jobs: c_int, ref global_upper_bound: atomic c_int,
		const times:c_ptr(c_int), const initial_depth: c_int,
    	ref node: fsp_node): (uint(64),uint(64)){


		var depth: c_int; //needs to be int because -1 is the break condition

		//one for each thread
		var front: [0.._MAX_MACHINES_] c_int;//private - search
		var back: [0.._MAX_MACHINES_] c_int;
		var remain: [0.._MAX_MACHINES_] c_int;

      	//state of the search
      	var control: [0.._MAX_JOBS_] bool = false;
		var scheduled: c_ptr(c_int);
      	var position: c_ptr(c_int);
      	var permutation: c_ptr(c_int);

		//aux
		var incumbent: c_int;
    	var lowerbound: c_int = 0;
    	var p1: c_int;

	    //CONTROL
		var num_sols: uint(64) = 0;
		var tree_size: uint(64) = 0;
		var metrics: (uint(64),uint(64));

		//SEARCH initialization
		scheduled = c_ptrTo(node.scheduled);
        position = c_ptrTo(node.position);
        permutation = c_ptrTo(node.permutation);
		for i in 0..jobs-1{
            control[i] = node.control[i];
        }
		depth = initial_depth;
    	//end of search init


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

                    lowerbound = simple_bornes_calculer(permutation, depth, jobs,
                         machines, jobs, c_ptrTo(remain), c_ptrTo(front), c_ptrTo(back),
                         minTempsArr_s, minTempsDep_s, times);

                    if(lowerbound<incumbent){

						control[scheduled[depth]] = true;
						depth +=1;
						tree_size+=1;

						if (depth == jobs && lowerbound<incumbent){
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
	}//end of node exploration


}
