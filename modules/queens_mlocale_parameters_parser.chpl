module queens_mlocale_parameters_parser{

	use Time;
	use CTypes;
	use checkpointing as checkpt;
	use DynamicIters;
	
	config param GPU: bool = false;

	if(GPU) then {
		use GPU_mlocale_utils;
		use queens_GPU_call_intermediate_search;
	}

	use queens_node_module;
	use queens_call_intermediate_search;
	use queens_tree_exploration;
	

	config param queens_checkPointer: bool = false;

	proc queens_mlocale_parameters_parser(
		const size: uint(16),
		const scheduler: string, const mode: string, const mlsearch:string ,
		const initial_depth: c_int, const second_depth: c_int,
		const lchunk: int, const mlchunk: int, const slchunk: int, const flag_coordinated: bool = false,
		ref distributed_active_set: [] queens_node, const Space: domain, ref metrics,
		ref tree_each_locale: [] uint(64),const pgas: bool, ref GPU_id: [] int, const CPUP: real,
		const language:string){

		const role_CPU: int = 1;
		const role_GPU: int = 0;




		if(queens_checkPointer){
			checkpt.start();
			begin checkpt.checkpointer(progress,partial_tree,synch_with_checkpointer,Space.size);
		//first task of the coordinator
		}////

		select scheduler{

			when "static" {
				if(!pgas){
					halt("### ERROR ERROR ###\n\t PGAS must be true for using static mlocale load distribution ### ");
				}

				select mlsearch{//what kind of multilocale search?

					when "naive"{
						
						writeln("################### Queens - distributed - naive - STATIC ###################");
						
						forall idx in distributed_active_set with (+ reduce metrics) do {
							
							var m1 = queens_subtree_explorer(size,initial_depth:int(32), idx);
		
							metrics+=m1;

							if(queens_checkPointer){
								checkpt.partial_tree.add(m1[1]);
								checkpt.partial_num_sol.add(m1[0]);
								checkpt.progress.add(1); //for each ite
							}//checkpointer
					
						}//for


					}//mlocale-naive

					
					when "mlocale"{

						forall n in distributed_active_set with (+ reduce metrics) do {
							var m1 = queens_call_intermediate_search(size,initial_depth,
								second_depth,slchunk,n,tree_each_locale);
							metrics+=m1;
							if(queens_checkPointer){
								checkpt.partial_tree.add(m1[1]);
								checkpt.partial_num_sol.add(m1[0]);
								checkpt.progress.add(1); //for each ite
							}//checkpointer

						}//for


					}//mlocale

					when "mlgpu"{
						if(GPU) then forall n in distributed_active_set with (+ reduce metrics) do  {
							var m1 = queens_GPU_call_intermediate_search(size,initial_depth,
								second_depth,slchunk,n,tree_each_locale, GPU_id[here.id],CPUP, mlsearch, language);
							metrics+=m1;
							if(queens_checkPointer){
								checkpt.partial_tree.add(m1[1]);
								checkpt.partial_num_sol.add(m1[0]);
								checkpt.progress.add(1);
							}//checkpointer
						}//for
					}//mlmgpu
				
					otherwise{
						 halt("###### ERROR ######\n###### ERROR ######\n###### ERROR ######\n###### WRONG PARAMETERS ######");
					}//
				}//mode
			}//end of STATIC







			when "dynamic" {
			
				select mlsearch{//what's the kind of multilocale search?


					when "naive"{ 
						writeln("################### Queens- distributed - naive - dynamic ###################");
						forall idx in distributedDynamic(c=Space, chunkSize=lchunk,localeChunkSize=mlchunk,coordinated=flag_coordinated) with (+ reduce metrics) do {
							
							var m1 = queens_subtree_explorer(size,initial_depth:int(32), distributed_active_set[idx]);
							
							metrics+=m1;

							if(queens_checkPointer){
								checkpt.partial_tree.add(m1[1]);
								checkpt.partial_num_sol.add(m1[0]);
								checkpt.progress.add(1);
							}//checkpointer

						}//for
					}//mlocale


					when "mlocale"{
						forall idx in distributedDynamic(c=Space, chunkSize=lchunk,localeChunkSize=mlchunk,coordinated=flag_coordinated) with (+ reduce metrics) do {
							var m1 = queens_call_intermediate_search(size,initial_depth,
								second_depth,slchunk,distributed_active_set[idx],tree_each_locale);
							metrics+=m1;
							if(queens_checkPointer){
								checkpt.partial_tree.add(m1[1]);
								checkpt.partial_num_sol.add(m1[0]);
								checkpt.progress.add(1);
							}//checkpointer

						}//for
					}//mlocale
					
					when "mlgpu"{
						if(GPU) then forall idx in distributedDynamic(c=Space, numTasks=1, chunkSize=lchunk,localeChunkSize=mlchunk,coordinated = flag_coordinated) with (+ reduce metrics) do {

							var m1 = queens_GPU_call_intermediate_search(size,initial_depth,
								second_depth,slchunk,distributed_active_set[idx],tree_each_locale,
								GPU_id[here.id], CPUP, mlsearch, language);	
								
							metrics+=m1;
							if(checkpointer){
								checkpt.partial_tree.add(m1[1]);
								checkpt.partial_num_sol.add(m1[0]);
								checkpt.progress.add(1);
							}//checkpointer
						}//for
					}//mlgpu
						
					when "dcpugpu"{
						if(GPU) then forall idx in distributedDynamic(c=Space,chunkSize=lchunk,localeChunkSize=mlchunk,coordinated=flag_coordinated) with (+ reduce metrics) do {

							var role: int = (here.id-flag_coordinated:int)%2;
							var m1: (uint(64),uint(64));
							if(role == role_GPU) {
								//writeln("Going on GPU:");
								m1 = queens_GPU_call_intermediate_search(size,initial_depth,
									second_depth, slchunk, distributed_active_set[idx], tree_each_locale,
									GPU_id[here.id], 0,mlsearch,language);
							}
							else{
								//writeln("Going on CPU:");
								m1 = queens_call_intermediate_search(size,initial_depth,
								second_depth-2,slchunk,distributed_active_set[idx],tree_each_locale);
							}

							metrics+=m1;
							if(queens_checkPointer){
								checkpt.partial_tree.add(m1[1]);
								checkpt.partial_num_sol.add(m1[0]);
								checkpt.progress.add(1);
							}//checkpointer

						}//for
					}//mlocale
					otherwise{
						 halt("###### ERROR ######\n###### ERROR ######\n###### ERROR ######\n###### WRONG PARAMETERS ######");
					}//
				}//mode

			}//end of dynamic
			when "guided" {

				select mlsearch{

					when "mlocale"{
						forall idx in distributedGuided(c=Space,minChunkSize=mlchunk,coordinated=flag_coordinated) with (+ reduce metrics) do {
							var m1 = queens_call_intermediate_search(size,initial_depth,
								second_depth,slchunk,distributed_active_set[idx],tree_each_locale);
							metrics+=m1;
							if(queens_checkPointer){
								checkpt.partial_tree.add(m1[1]);
								checkpt.partial_num_sol.add(m1[0]);
								checkpt.progress.add(1);
	                		}//checkpointer
						 }//
					}//mlocale
				
					
						when "mlgpu"{
							if(GPU) then forall idx in distributedGuided(c=Space,minChunkSize=mlchunk,coordinated=flag_coordinated) with (+ reduce metrics) do {
								var m1 = queens_GPU_call_intermediate_search(size,initial_depth,
									second_depth,slchunk,distributed_active_set[idx],tree_each_locale,
									GPU_id[here.id],CPUP,mlsearch,language);
								metrics+=m1;
								if(queens_checkPointer){
									checkpt.partial_tree.add(m1[1]);
									checkpt.partial_num_sol.add(m1[0]);
									checkpt.progress.add(1);
								}//checkpointer
							}//

						}//mlmgpu
					
					otherwise{
						halt("###### ERROR ######\n###### ERROR ######\n###### ERROR ######\n###### WRONG PARAMETERS ######");
					}//
				}//mode
			}//guided

			otherwise{
				halt("###### ERROR ######\n###### ERROR ######\n###### ERROR ######\n###### WRONG PARAMETERS ######");
			}//otherwise

		}//scheduler

		if(queens_checkPointer) then checkpt.wait();

	}//

}
