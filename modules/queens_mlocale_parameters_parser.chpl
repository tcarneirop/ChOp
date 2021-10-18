module queens_mlocale_parameters_parser{

	use Time;
	use SysCTypes;
	use checkpointing as checkpt;
	use DynamicIters;
	use GPU_mlocale_utils;
	use queens_node_module;
	use queens_call_intermediate_search;
	use queens_GPU_call_intermediate_search;

	config param queens_checkPointer: bool = false;

	proc queens_mlocale_parameters_parser(
		const size: uint(16),
		const scheduler: string, const mode: string, const mlsearch:string ,
		const initial_depth: c_int, const second_depth: c_int,
		const lchunk: int, const mlchunk: int, const slchunk: int, const flag_coordinated: bool = false,
		ref distributed_active_set: [] queens_node, const Space: domain, ref metrics,
		ref tree_each_locale: [] uint(64),const pgas: bool, ref GPU_id: [] int, const CPUP: real){

		
		writeln("###### QUEENS IMPROVED MLOCALE ######");
		
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

					when "mlocale"{
						forall n in distributed_active_set with (+ reduce metrics) do {
							var m1 = queens_call_intermediate_search(size,initial_depth,
								second_depth,slchunk,n,tree_each_locale);
							metrics+=m1;
							if(queens_checkPointer){
								checkpt.partial_tree.add(m1[1]);
	                			checkpt.progress.add(1);
	                		}//checkpointer
	                		
						}//for
					}//mlocale
					when "mlgpu"{
						//@todo -- IT IS GOING TO CHANGE
						forall n in distributed_active_set with (+ reduce metrics) do  {
							var m1 = queens_GPU_call_intermediate_search(size,initial_depth,
								second_depth,slchunk,n,tree_each_locale, GPU_id[here.id],CPUP);
							metrics+=m1;
							if(queens_checkPointer){
								checkpt.partial_tree.add(m1[1]);
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
					when "mlocale"{
						forall idx in distributedDynamic(c=Space,chunkSize=lchunk,localeChunkSize=mlchunk,coordinated=flag_coordinated) with (+ reduce metrics) do {
							var m1 = queens_call_intermediate_search(size,initial_depth,
								second_depth,slchunk,distributed_active_set[idx],tree_each_locale);
							metrics+=m1;
							if(queens_checkPointer){
								checkpt.partial_tree.add(m1[1]);
	                			checkpt.progress.add(1);
	                		}//checkpointer

						}//for
					}//mlocale

					when "mlgpu"{
						forall idx in distributedDynamic(c=Space,chunkSize=lchunk,localeChunkSize=mlchunk,coordinated = flag_coordinated) with (+ reduce metrics) do {
							var m1 = queens_GPU_call_intermediate_search(size,initial_depth,
								second_depth,slchunk,distributed_active_set[idx],tree_each_locale,
								GPU_id[here.id], CPUP);
							metrics+=m1;
							if(checkpointer){
								checkpt.partial_tree.add(m1[1]);
	                			checkpt.progress.add(1);
	                		}//checkpointer
						}//for  
					}//mlmgpu
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
	                			checkpt.progress.add(1);
	                		}//checkpointer
						 }//
					}//mlocale
					when "mlmgpu"{
						forall idx in distributedGuided(c=Space,minChunkSize=mlchunk,coordinated=flag_coordinated) with (+ reduce metrics) do {
							var m1 = queens_GPU_call_intermediate_search(size,initial_depth,
								second_depth,slchunk,distributed_active_set[idx],tree_each_locale,
								GPU_id[here.id],CPUP);
							metrics+=m1;
							if(queens_checkPointer){
								checkpt.partial_tree.add(m1[1]);
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
