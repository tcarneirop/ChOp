module fsp_node_module{

	//use fsp_simple_chpl_c_headers;
	use fsp_constants;
	use SysCTypes;
	
	record fsp_node{

		var scheduled: [0.._MAX_JOBS_] c_int;
	    var position: [0.._MAX_JOBS_] c_int;
	    var permutation: [0.._MAX_JOBS_] c_int;
	    var control: [0.._MAX_JOBS_] bool;
	
	}

	//// record fsp_read_only_vectors{
	// 	var minTempsDep: [0.._MAX_MACHINES_] c_int;//read only
	// 	var minTempsArr: [0.._MAX_MACHINES_] c_int; //read only
	// }


}//end of module