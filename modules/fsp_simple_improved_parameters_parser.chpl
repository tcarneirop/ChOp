module fsp_simple_improved_parameters_parser{


	use Time;
    use CTypes;
    use checkpointing;
    use DistributedIters;
    use fsp_node_module;
	use fsp_simple_call_improved_mlocale_search;

	//@todo:
	config param simple_checkPointer: bool = false;

	proc fsp_simple_improved_mlocale_parameters_parser(const atmc_type: string,
    	const scheduler: string, const machines: c_int,const jobs: c_int,
    	const initial_depth: c_int, const second_depth: c_int,
    	const lchunk: int, const mlchunk: int, const slchunk: int, const flag_coordinated: bool = false,
    	ref distributed_active_set: [] fsp_node, ref set_of_atomics: [] atomic c_int,
    	ref global_ub: atomic c_int, const Space: domain, ref metrics: (uint(64),uint(64)),
    	ref tree_each_locale: [] uint(64),const pgas: bool){

		writeln("###### SIMPLE IMPROVED MLOCALE ######");
    	writeln("\n ### TODO: No atomics ###\n");

	    var ub: c_int = global_ub.read();

	  	//@todo -s if checkpointer
	  	var progress: atomic uint(64);
	  	var partial_tree: atomic uint(64);
	  	var synch_with_checkpointer: atomic bool;
	  	//progress bar and checkpointing
	  	progress.write(0:uint(64));
	  	partial_tree.write(0:uint(64));
	  	synch_with_checkpointer.write(false);


	  	begin checkpointer(progress,partial_tree,synch_with_checkpointer,Space.size); //first task of the coordinator
	  	//@endtodo -s if checkpointer


		select scheduler{//second task of the coordinator
	        when "static" {
	        	if(!pgas){
	        		halt("### ERROR ERROR ### \n### ERROR ERROR ###\n\t### PGAS must be true for using static mlocale load distribution ### ");
	        	}
	            forall n in distributed_active_set with (+ reduce metrics) do {
	                metrics+=fsp_simple_call_improved_mlocale_search(machines,jobs,initial_depth,
	                 	second_depth,slchunk,n,ub,tree_each_locale);
	            }
	        }//STATIC
	        when "dynamic" {
	            forall idx in distributedDynamic(c=Space,chunkSize=lchunk,localeChunkSize=mlchunk,coordinated = flag_coordinated)
	             with (+ reduce metrics, var local_metrics: (uint(64),uint(64))) do {
	            	//, var local_metrics: (uint(64),uint(64))
	            	//writeln("Lets begin!");
	                local_metrics+=fsp_simple_call_improved_mlocale_search(machines,jobs,initial_depth,
	                 	second_depth,slchunk,distributed_active_set[idx],ub,tree_each_locale);
	                //writeln("done!");
	                //@todo -s if checkpointer
	                partial_tree.add(local_metrics[1]);
	                progress.add(1);
	                //@endtodo -s if checkpointer

	                metrics+=local_metrics;

	            }
	        }//dynamic
	        when "guided" {
	        	forall idx in distributedGuided(c=Space, minChunkSize=lchunk, coordinated = flag_coordinated) with (+ reduce metrics) do {
	                 metrics+=fsp_simple_call_improved_mlocale_search(machines,jobs,initial_depth,
	                 	second_depth,slchunk,distributed_active_set[idx],ub,tree_each_locale);
	            }
	        }//guided
	        otherwise{
	            halt("###### ERROR ######\n###### ERROR ######\n###### ERROR ######\n###### WRONG PARAMETERS ######");
	        }

	    } //select

	    //@todo -s if checkpointer
		var end_of_search: bool = synch_with_checkpointer.read();

	    while(!end_of_search){
	    	sleep(1);
	    	end_of_search = synch_with_checkpointer.read();
	    }//while
	    //@endtodo -s if checkpointer

    }//call search




}///////////////////////////
