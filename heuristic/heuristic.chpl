
use Random;



var num_depths : int = 3;
var num_schedulers: int = 3;
var num_chunks: int = 5; 
var num_max_threads: int = 4;
var max_threads = 48;

var num_par = 8;

var num_par_vec: [0..#num_par] int = [
			num_schedulers,
			num_depths, 
			num_depths, 
			num_chunks, 
			num_chunks,
			num_max_threads, 
			2,			
			2];


var initial_depth_vector: [0..#num_depths] int = [3,4,5];
var second_depth_vector: [0..#num_depths] int  = [6,7,8];
var scheduler_vector: [0..#num_schedulers] string = ["static", "dynamic", "guided"];
var mlchunk_vector: [0..#num_chunks] int = [1,4,8,16,32];
var slchunk_vector: [0..#num_chunks] int = [8,16,32,64,128];

var coordinated_vector: [0..1] bool = [true,false];

var pgas_vector: [0..1] bool = [true,false];

var num_threads_vector: [0..#num_max_threads] int  = [max_threads,max_threads/2,max_threads/3,max_threads/4];





record Solution{
	
	var inner_organization: (int,int, int, int, int, int, int, int);
	var parameters: (string,int, int, int, int, int, bool, bool);
	var cost: int = 0;
	
	proc init(){

		var randStream = new owned RandomStream(int);
		
		inner_organization = (
			abs(randStream.getNext()) % (num_schedulers),
			abs(randStream.getNext()) % (num_depths), 
			abs(randStream.getNext()) % (num_depths), 
			abs(randStream.getNext()) % (num_chunks), 
			abs(randStream.getNext()) % (num_chunks),
			abs(randStream.getNext()) % (num_max_threads), 
			abs(randStream.getNext()) % (2),			
			abs(randStream.getNext()) % (2)
		);

		parameters =  (
			scheduler_vector[inner_organization(0)],
			initial_depth_vector[inner_organization(1)], 
			second_depth_vector[inner_organization(2)], 
			mlchunk_vector[inner_organization(3)], 
			slchunk_vector[inner_organization(4)],
			num_threads_vector[inner_organization(5)], 
			coordinated_vector[inner_organization(6)],
			pgas_vector[inner_organization(7)]
		);
		this.complete();
		setCost();
	}//init()


	proc init(neighbor_inner_organization: (int,int, int, int, int, int, int, int)){
		inner_organization = neighbor_inner_organization;
		parameters =  (
			scheduler_vector[inner_organization(0)],
			initial_depth_vector[inner_organization(1)], 
			second_depth_vector[inner_organization(2)], 
			mlchunk_vector[inner_organization(3)], 
			slchunk_vector[inner_organization(4)],
			num_threads_vector[inner_organization(5)], 
			coordinated_vector[inner_organization(6)],
			pgas_vector[inner_organization(7)]
		);
		this.complete();
		setCost();
	}//init()


	proc neighbor(const par: int, const right: bool, const left: bool):Solution{

		var neighbor_inner_organization = inner_organization;

		if(right){

			var neighbor_par_right = inner_organization(par) + 1;

			if(neighbor_par_right < num_par_vec[par]){
				neighbor_inner_organization(par) = neighbor_par_right;
				//writeln("\n ####### FUNCTION neighbor parameter right", par, " : ", neighbor_inner_organization);
			}else{
 				//writeln("####### FUNCTION No Right neighbor for parameter ", par);
 			}
	

		}

		if(left){
			var neighbor_par_left = inner_organization(par) - 1;
		
			if(neighbor_par_left >= 0){
				neighbor_inner_organization(par) = neighbor_par_left;
				//writeln("\n ####### FUNCTION neighbor parameter left ", par, " : ", neighbor_inner_organization);
			}else{
 				//writeln("####### FUNCTION No Left neighbor for parameter ", par);
 			}
	
		}
	
		var neighbor = new Solution(neighbor_inner_organization);
		return neighbor;
	}///////

	proc setCost(){
		cost = 0;
		for i in 0..#num_par do{
			cost+=inner_organization(i);
		}
	}
	proc getCost():int{
		return cost;
	}
}//solution


var initial_solution = new Solution();

local_search(initial_solution);

proc hillClimb(initial_solution){




}

proc local_search(ref initial_solution: Solution, const first_improvement: bool = false): Solution{

	var qtd_neighbor: int = 0;
	
	var ls_cost: int = 0;
	
	var newSolRight: Solution;
	
	var newSolLeft: Solution;
	
	var best_sol: Solution = initial_solution;

	var best_neighbor: Solution;
	
	var min_sol_cost:int = initial_solution.getCost();

	writeln("\n\n#####################");
	writeln("Initial sol: \n", initial_solution);

	for par in 0..#num_par do{


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

	 	if(newSolRight.getCost() < newSolLeft.getCost()){ 
	 		best_neighbor = newSolRight; 
	 	}else{
	 		best_neighbor = newSolLeft;
	 	}

	 	if(best_neighbor.getCost()< best_sol.getCost()){
	 		writeln("\n\n### improvement found: from ", best_sol.getCost(), " to ", best_neighbor.getCost() );
	 		best_sol = best_neighbor;

	 	}
		
	 }

	writeln("\n\n#####################");
	writeln("Initial Solution: \n",initial_solution);

	writeln("\n\n#####################");
	writeln("Local search solution: \n", best_sol);

	return best_sol;

}







// var i: int = 0;
// while(i<100){

// 	var sol = new Solution();
// 	writeln(sol.parameters);
// 	writeln(sol.inner_organization);
// 	writeln("\n");
// 	i+=1;
// }




// writeln("New sol: \n", sol);


// for par in 0..#num_par do{

//  	var sol = new Solution();
//  	var neighbor_par_right = sol.inner_organization(par) + 1;
//  	writeln("\n\n#####################");
//  	writeln("New sol: \n", sol);

//  	if(neighbor_par_right < num_par_vec[par]){
 	
//  		writeln("\n neighbor parameter right: ", par, " \n");
//  		var newSolRight = sol.neighbor(par,true,false);
//  		writeln("\n neighbor right:", newSolRight);
//  	}else{
//  		writeln("No Right neighbor for parameter ", par);
//  	}

//  	var neighbor_par_left = sol.inner_organization(par) - 1;
 	
//  	if(neighbor_par_left >= 0){
//  		writeln("\n Neighbor parameter Left: ", par, " \n");
//  		var newSolLeft = sol.neighbor(par,false,true);
//  		writeln("\nNeighbor Left:", newSolLeft);
//  	}
//  	else{
//  		writeln("\nNo Left neighbor for parameter ", par);
//  	}
	
//  }

//lchunk = (1)

//----  3*3*3*5*5*2*2 (2700)