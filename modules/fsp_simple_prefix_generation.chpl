
module fsp_simple_prefix_generation{

    use fsp_simple_chpl_c_headers;
    use fsp_node_module;
    use fsp_constants;
    use DynamicIters;
    use SysCTypes;
    use CPtr;    
    

    proc fsp_simple_prefix_generation(const machines: c_int, const jobs: c_int, upper_bound: c_int , 
      const times:c_ptr(c_int), const initial_depth: c_int,  set_of_nodes: [] fsp_node):  (uint(64),uint(64)){

      var depth: c_int = 0; //needs to be int because -1 is the break condition

      var front: [0.._MAX_MACHINES_] c_int;//private - search
      var back: [0.._MAX_MACHINES_] c_int;
      var remain: [0.._MAX_MACHINES_] c_int;

      //state of the search
      var scheduled: [0.._MAX_JOBS_] c_int = _FSP_EMPTY_;
      var position: [0.._MAX_JOBS_] c_int =  [i in 0.._MAX_JOBS_] i;
      var permutation: [0.._MAX_JOBS_] c_int = [i in 0.._MAX_JOBS_] i;
      var control: [0.._MAX_JOBS_] bool = false;

      //aux 
      var incumbent: c_int = upper_bound;
      var lowerbound: c_int = 0;
      var p1: c_int;

        //CONTROL
      var num_prefixes: uint(64) = 0;
      var tree_size: uint(64) = 0;
      var metrics: (uint(64),uint(64));

      //fsp init
      //start_vector(c_ptrTo(position),jobs);
      //start_vector(c_ptrTo(permutation),jobs);

      depth = 0;
      

      while(true){//Search

        scheduled[depth] = scheduled[depth]+1;

        if scheduled[depth] == jobs then
          scheduled[depth] = _FSP_EMPTY_;
        else{
          if (!control[scheduled[depth]]) {//valid

            p1 = permutation[depth];
            swap(permutation[depth],permutation[position[scheduled[depth]]]);
            swap(position[scheduled[depth]],position[p1]);

            lowerbound = simple_bornes_calculer(c_ptrTo(permutation), depth, jobs,
                         machines, jobs, c_ptrTo(remain), c_ptrTo(front), c_ptrTo(back), 
                         minTempsArr_s, minTempsDep_s, times);

            if(lowerbound<incumbent){//and feasible

              control[scheduled[depth]] = true;
              depth +=1;
              tree_size+=1;

              if (depth == initial_depth){ //and complete
                  //let's parallelize it
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

      if (depth < 0) then
        break;
    }//search

    metrics[0] = num_prefixes;
    metrics[1] = tree_size;
    return metrics;
  }//end of prefix gen

}//module