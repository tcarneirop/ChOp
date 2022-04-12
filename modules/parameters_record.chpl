module parameters_record{

	use CTypes;

	record commandline_parameters{

		var initial_depth: c_int;
		var second_depth:  c_int;
		var size: uint(16); //queens

		//the default coordinated is TRUE
		var scheduler: string;

		var mlchunk: int; //inter-node chunk.
		var lchunk: int; //task chunk the inter-node scheduler gives.
		var slchunk: int; //chunk for the second level of parallelism.

		var coordinated: bool;  //centralized node?
		var mode: string; //keep it improved, i.e., second level of parallelism by hand.
		var pgas: bool; //pgas-based active set?

		var num_threads: int; //number of threads.
		var profiler: bool; //to gather profiler metrics and execution graphics.
		var number_exec: int;   //going to be removed soon.
		var upper_bound: c_int; //value for the initial upper bound. If it is zero, the optimal solution is going to be used.
		var lower_bound: string; //type of lowerbound. Johnson and simple.
		var atype: string; //atomic type. 'none' when initializing using the optimal -- use like that.
		var instance: int(8); //fsp instance
		var verbose: bool; //verbose network communication

		proc init() {

			initial_depth = 4;
			second_depth  = 7;
			size          = 12; //queens

			//the default coordinated is TRUE
			scheduler = "dynamic";
			mlchunk = 0; //inter-node chunk, default
			lchunk  = 1; //task chunk the inter-node scheduler gives.
			slchunk = 4; //chunk for the second level of parallelism.

			coordinated= true;  //centralized node?
			mode= "improved"; //keep it improved, i.e., second level of parallelism by hand
			pgas = false; //pgas-based active set?

			num_threads= here.maxTaskPar; //number of threads.
			profiler= false; //to gather profiler metrics and execution graphics.
			number_exec = 1;   //going to be removed soon.
			upper_bound = 0; //value for the initial upper bound. If it is zero, the optimal solution is going to be used.
			lower_bound = "johnson"; //type of lowerbound. Johnson and simple.
			atype = "none"; //atomic type. 'none' when initializing using the optimal -- use like that!!
			instance = 13; //fsp instance
			verbose = false; //verbose network communication
		}////


		proc init(initial_depth: c_int, second_depth:  c_int, size: uint(16), scheduler: string,
				mlchunk: int, lchunk: int, slchunk: int, coordinated: bool, mode: string, pgas: bool,
				num_threads: int, profiler: bool, number_exec: int, upper_bound: c_int, lower_bound: string,
				atype: string, instance: int(8), verbose: bool = false){

			this.initial_depth = initial_depth;
			this.second_depth  = second_depth;
			this.size          = size; //queens

			//the default coordinated is TRUE
			this.scheduler = scheduler;
			this.mlchunk = mlchunk; //inter-node chunk, default
			this.lchunk  = lchunk; //task chunk the inter-node scheduler gives.
			this.slchunk = slchunk; //chunk for the second level of parallelism.

			this.coordinated = coordinated;  //centralized node?
			this.mode = mode; //keep it improved, i.e., second level of parallelism by hand
			this.pgas = pgas; //pgas-based active set?

			this.num_threads = num_threads; //number of threads.
			this.profiler = profiler; //to gather profiler metrics and execution graphics.
			this.number_exec = number_exec;   //going to be removed soon.
			this.upper_bound = upper_bound; //value for the initial upper bound. If it is zero, the optimal solution is going to be used.
			this.lower_bound = lower_bound; //type of lowerbound. Johnson and simple.
			this.atype = atype; //atomic type. 'none' when initializing using the optimal -- use like that!!
			this.instance = instance; //fsp instance
			this.verbose = verbose; //verbose network communication
		}//
	}//record

}////////////////
