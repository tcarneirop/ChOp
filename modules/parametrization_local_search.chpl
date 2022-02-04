module parametrization_local_search{

	use parametrization_solution;
	var num_solutions: int = 0;

	proc initialization(const heuristic: string, const problem: string = "simple", const instance: int, const mode: string = "mcore" ){
		
		writeln("\n\n##################### creating the first solution");
		var initial_solution = new Solution(problem,instance);
		num_solutions+=1;

		if(heuristic == "hc"){
			hillClimb(initial_solution);
		}
		else{
			local_search(initial_solution);
		}
	}


	proc hillClimb(initial_solution){

		writeln("\n\n##################### hillClimb");

		var cost: real = initial_solution.getCost();
		var new_cost: real = 0.0;
		var workingSolution: Solution = initial_solution;
		var newSolution: Solution;
		var iteration = 0;

		writeln("Initial sol: \n", initial_solution);

		new_cost = 0.0;

		while(new_cost<cost){
			writeln("### Hillclimb iteration: ", iteration,"\n\tWorking cost: ", cost,"\n\tNew cost: ", new_cost);
			cost = workingSolution.getCost();
			newSolution = local_search(workingSolution);
			new_cost = newSolution.getCost();
			workingSolution = newSolution;
		}

		writeln("\n\n##################### hillClimb - Initial Sol");
		writeln("Initial sol: \n", initial_solution);

		writeln("\n\n##################### hillClimb - Final Sol");
		writeln("Final sol: \n", workingSolution);

	}

	proc local_search(ref initial_solution: Solution, const first_improvement: bool = false): Solution{

		writeln("\n\n#### Local Search ####");

		var qtd_neighbor: int = 0;
		
		var ls_cost: real = 0.0;
		
		//var newSolRight: Solution;

		var newSolRight: Solution;
		var newSolLeft: Solution;
		var best_neighbor: Solution;

		var best_sol: Solution = initial_solution;
		
		var min_sol_cost: real = initial_solution.getCost();

		writeln("\n\n#####################");
		writeln("Initial sol: \n", initial_solution);

		for par in 0..#max_par do{

		 	var neighbor_par_right = initial_solution.inner_organization(par) + 1;

		 	if(neighbor_par_right < num_par_vec[par]){
		 	
		 		//writeln("\n neighbor parameter right: ", par, " \n");
		 		newSolRight = initial_solution.neighbor(par,true,false);
		 		//writeln("\n neighbor right:", newSolRight);
		 	}else{
		 		writeln("No Right neighbor for parameter ", par);
		 	}

		 	var neighbor_par_left = initial_solution.inner_organization(par) - 1;
		 	
		 	if(neighbor_par_left >= 0){
		 		//writeln("\n Neighbor parameter Left: ", par, " \n");
		 		newSolLeft = initial_solution.neighbor(par,false,true);
		 		//writeln("\nNeighbor Left:", newSolLeft);
		 	}
		 	else{
		 		writeln("\nNo Left neighbor for parameter ", par);
		 	}

		 	//@todo -- min()

		 	if(newSolRight.getCost() < newSolLeft.getCost()){ 
		 		best_neighbor = newSolRight; 
		 	}else{
		 		best_neighbor = newSolLeft;
		 	}

		 	//@todo -- min()
		 	
		 	if(best_neighbor.getCost() < best_sol.getCost()){
		 		writeln("\n\n#################################### improvement found: from ", best_sol.getCost(), " to ", best_neighbor.getCost() );
		 		best_sol = best_neighbor;

		 	}
			
		 }

		writeln("\n\n#####################");
		writeln("Initial Solution: \n",initial_solution);

		writeln("\n\n#####################");
		writeln("Local search solution: \n", best_sol);

		return best_sol;

	}



}