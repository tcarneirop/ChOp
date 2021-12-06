
module parametrization_solution{

	use Random;
	use SysCTypes;
	use Time;
    use CPtr;
	use fsp_johnson_call_mcore_search;
	use fsp_simple_call_mcore_search;

	var num_depths : int = 3;
	var num_schedulers: int = 3;
	var num_blocks: int = 5;
	var num_chunks: int = 5; 
	var num_max_threads: int = 4;
	var max_threads = here.maxTaskPar;
	var num_par: int;

	var max_par = 8;

	var num_par_vec: [0..#max_par] int = [
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
	var block_size_vector: [0..#num_blocks] int = [64,128,196,256,512];

	var coordinated_vector: [0..1] bool = [true,false];

	var pgas_vector: [0..1] bool = [true,false];

	var num_threads_vector: [0..#num_max_threads] int  = [max_threads,max_threads/2,max_threads/3,max_threads/4];


	record Solution{
		
		var inner_organization: (int,int, int, int, int, int, int, int);
		var parameters: (string,int, int, int, int, int, bool, bool);
		var cost_tuple: (real,real,real);
		var cost: real = 0.0;
		
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
			cost_tuple = fsp_simple_call_multicore_search(parameters(1):c_int,0:c_int,parameters(0),
				parameters(4),parameters(5),13,false);
			cost = cost_tuple(2);
			
		}

		proc getCost():real{
			return cost;
		}

		proc getCostTuple():(real,real,real){
			return cost_tuple;
		}
	}//solution

}//module