module fsp_johnson_serial{

	//C header
	use fsp_johnson_chpl_c_headers;
	use fsp_constants;
    use fsp_aux;
	use SysCTypes;
	use Time;
    use CPtr;

	
	proc fsp_johnson_call_serial(upper_bound: c_int = _FSP_INF_,const instance: c_short){

		var timer: Timer;
		var jobs: c_int;
    	var machines: c_int;
    	var metrics: (uint(64),uint(64),c_int);
    	var times: c_ptr(c_int) = get_instance(machines,jobs, instance);

    	writeln("Machines: ", machines);
    	writeln("Jobs: ", jobs);
    	print_instance(machines,jobs,times);

  		timer.start();
    	metrics = fsp_johnson_serial(machines,jobs,upper_bound,times);
    	timer.stop(); 
    	
        fsp_print_serial_report(timer, machines, jobs, metrics, upper_bound);

    	timer.clear();
  
	}//Call serial search



    proc fsp_johnson_serial(const machines: c_int, const jobs: c_int, upper_bound: c_int , const times:c_ptr(c_int) ):  (uint(64),uint(64),c_int){

                
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
        var num_sols: uint(64) = 0;
        var tree_size: uint(64) = 0;
        var metrics: (uint(64),uint(64),c_int);

        //fsp init
        johnson_remplirMachine(machines, machine);
        remplirTempsArriverDepart(minTempsArr,minTempsDep, machines, jobs, times);
        johnson_remplirLag(machines, jobs, machine, tempsLag,times);
        johnson_remplirTabJohnson(machines, jobs, tabJohnson, tempsLag, times);


        depth = 0;

        while(true){//Search

            scheduled[depth] = scheduled[depth]+1;

            if scheduled[depth] == jobs then
                scheduled[depth] = _FSP_EMPTY_;
            else{
                if (!control[scheduled[depth]]) {

                    p1 = permutation[depth];
                    swap(permutation[depth],permutation[position[scheduled[depth]]]);
                    swap(position[scheduled[depth]],position[p1]);

                    lowerbound = johnson_bornes_calculer(machines, jobs, c_ptrTo(job), c_ptrTo(permutation),
                        depth, jobs, incumbent, c_ptrTo(tempsMachines), c_ptrTo(tempsMachinesFin), minTempsArr,
                        minTempsDep, machine, tempsLag,times);

                    if(lowerbound<incumbent){

                        control[scheduled[depth]] = true;
                        depth +=1;
                        tree_size+=1;

                        if (depth == jobs && lowerbound < incumbent){           
                            num_sols+=1;
                            incumbent = lowerbound;
                            writeln("\nIncumbent of number ", num_sols, " found.\n\t","Cost: " ,incumbent, "\n");
                            //print_permutation(c_prtTo(permutation),jobs);
                        }
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

        metrics[0] = num_sols;
        metrics[1] = tree_size;
        metrics[2] = incumbent;

        return metrics;
    }//end of regular


}//end module