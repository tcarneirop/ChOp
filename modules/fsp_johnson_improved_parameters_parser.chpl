module fsp_johnson_improved_parameters_parser{

	use List;
	use Time;
	use CTypes;
    use DistributedIters;
    use fsp_node_module;
	use fsp_johnson_call_improved_mlocale_search;

	//@TODO: improving here to use localeChunk and coordinated
    proc fsp_johnson_improved_parameters_parser(
    	const atmc_type: string,
    	const scheduler: string, const machines: c_int,const jobs: c_int,
    	const initial_depth: c_int, const second_depth: c_int,
    	const lchunk: int, const mlchunk: int, const slchunk: int , const flag_coordinated: bool = false,
    	ref distributed_active_set: [] fsp_node, ref set_of_atomics: [] atomic c_int,
    	ref global_ub: atomic c_int, const Space: domain, ref metrics: (uint(64),uint(64)),
    	ref tree_each_locale: [] uint(64), const pgas: bool ){

    	writeln("###### JOHNSON IMPROVED MLOCALE ######");

	    var ub: c_int = global_ub.read();

	    select scheduler{
	        when "static" {
	        	if(!pgas){
	        		halt("### ERROR ERROR ### \n### ERROR ERROR ###\n\t### PGAS must be true for using static mlocale load distribution ### ");
	        	}
	            forall n in distributed_active_set with (+ reduce metrics) do {
	                metrics+=fsp_johnson_call_improved_mlocale_search(machines,jobs,initial_depth,
	                 	second_depth,slchunk,n,ub,tree_each_locale);
	            }

	        }//STATIC
	        when "dynamic" {
	            forall idx in distributedDynamic(c=Space,chunkSize=lchunk,localeChunkSize=mlchunk,coordinated = flag_coordinated) with (+ reduce metrics) do {
	                metrics+=fsp_johnson_call_improved_mlocale_search(machines,jobs,initial_depth,
	                 	second_depth,slchunk,distributed_active_set[idx],ub,tree_each_locale);
	            }

	        }//dynamic
	        when "guided" {
	             forall idx in distributedGuided(c=Space, minChunkSize=lchunk, coordinated = flag_coordinated) with (+ reduce metrics) do {
	                  metrics+=fsp_johnson_call_improved_mlocale_search(machines,jobs,initial_depth,
	                  	second_depth,slchunk,distributed_active_set[idx],ub,tree_each_locale);
	             }
	        }//guided
	        otherwise{
	            halt("###### ERROR ######\n###### ERROR ######\n###### ERROR ######\n###### WRONG PARAMETERS ######");
	        }

	    }


    }//call search

}//////module
