
module fsp_johnson_prefix_generation{

    use fsp_johnson_chpl_c_headers;
    use fsp_node_module;
    use fsp_constants;
    use CTypes;
    //use CPtr;


    proc fsp_johnson_prefix_generation(const machines: c_int, const jobs: c_int, upper_bound: c_int ,
      const times:c_ptr(c_int), const initial_depth: c_int,  set_of_nodes: [] fsp_node):  (uint(64),uint(64)){

      var depth: c_int = 0; //needs to be int because -1 is the break condition

      var tempsMachinesFin: [0.._MAX_MCHN_] c_int; //front
      var tempsMachines: [0.._MAX_MCHN_] c_int; //back
      var job: [0.._MAX_J_JOBS_] c_int;

        //state of the search
      var scheduled: [0.._MAX_J_JOBS_] c_int = _FSP_EMPTY_;
      var position: [0.._MAX_J_JOBS_] c_int =  [i in 0.._MAX_J_JOBS_] i;
      var permutation: [0.._MAX_J_JOBS_] c_int = [i in 0.._MAX_J_JOBS_] i;
      var control: [0.._MAX_J_JOBS_] bool = false;

      //aux
      var incumbent: c_int = upper_bound;
      var lowerbound: c_int = 0;
      var p1: c_int;

      //CONTROL
      var num_prefixes: uint(64) = 0;
      var tree_size: uint(64) = 0;
      var metrics: (uint(64),uint(64));

        // //fsp init -- let's put outside
        // johnson_remplirMachine(machines, machine);
        // remplirTempsArriverDepart(minTempsArr,minTempsDep, machines, jobs, times);
        // johnson_remplirLag(machines, jobs, machine, tempsLag,times);
        // johnson_remplirTabJohnson(machines, jobs, tabJohnson, tempsLag, times);

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

            lowerbound = johnson_bornes_calculer(machines, jobs, c_ptrTo(job), c_ptrTo(permutation),
                depth, jobs, incumbent, c_ptrTo(tempsMachines), c_ptrTo(tempsMachinesFin), minTempsArr,
                minTempsDep, machine, tempsLag,times);

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
      if (depth < 0) then
        break;
      control[scheduled[depth]] = false;
    }//search

    metrics[0] = num_prefixes;
    metrics[1] = tree_size;
    return metrics;
  }//end of prefix gen


}//module
