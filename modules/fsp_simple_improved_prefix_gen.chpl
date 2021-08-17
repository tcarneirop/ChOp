module fsp_simple_improved_prefix_gen{
 	
 	use fsp_simple_chpl_c_headers;
    use fsp_node_module;
    use fsp_constants;
    use SysCTypes;
    

	proc fsp_simple_improved_prefix_gen(const machines: c_int, const jobs: c_int, 
		const initial_depth: c_int, const second_depth: c_int, ref node: fsp_node, set_of_nodes: [] fsp_node,
		global_upper_bound:c_int ): (uint(64),uint(64)){

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
		var num_prefixes: uint(64) = 0;
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
                         minTempsArr, minTempsDep, c_temps);

                    if(lowerbound<incumbent){

						control[scheduled[depth]] = true;
						depth +=1;
						tree_size+=1;

		              if (depth == second_depth){ //and complete
		
		                  for i in 0..jobs-1 do{
		                    set_of_nodes[num_prefixes].scheduled[i] = scheduled[i];
		                    set_of_nodes[num_prefixes].position[i] = position[i];
		                    set_of_nodes[num_prefixes].permutation[i] = permutation[i];
		                    set_of_nodes[num_prefixes].control[i] = control[i];
		                  }
		                  num_prefixes+=1;             
		              }//prefix copy
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

		metrics[0] = num_prefixes;
    	metrics[1] = tree_size;
		return metrics;

	}//node explorer

}