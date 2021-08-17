module fsp_johnson_mlocale_parameters_parser{


	use Time;
    use SysCTypes;
    use fsp_node_module;
    use fsp_johnson_multilocale_node_explorer;


    proc fsp_johnson_mlocale_parameters_parser(const atmc_type: string, const scheduler: string, const machines: c_int,const jobs: c_int ,const initial_depth: c_int ,
    	const chunk: int, ref distributed_active_set: [] fsp_node, ref set_of_atomics: [] atomic c_int, ref global_ub: atomic c_int,
    	const Space: domain, ref metrics: (uint(64),uint(64))){

	    select atmc_type{

	        when "global"{

	            writeln("\n### Global atomic upper bound on locale 0 ###\n");

	            select scheduler{
	                when "static" {
	                    forall n in distributed_active_set with (+ reduce metrics) do {
	                       metrics += fsp_johnson_mlocale_global_atomic_node_explorer(machines,jobs,initial_depth,
	                            n, global_ub);
	                    }

	                }//static
	                when "dynamic" {
	                    forall idx in distributedDynamic(c=Space, chunkSize=chunk) with (+ reduce metrics) do {
	                        metrics += fsp_johnson_mlocale_global_atomic_node_explorer(machines,jobs,initial_depth,
	                            distributed_active_set[idx], global_ub); 
	                    }
	                }//dynami
	                when "guided" {      
	                    forall idx in distributedDynamic(c=Space, chunkSize=chunk) with (+ reduce metrics) do {
	                        metrics += fsp_johnson_mlocale_global_atomic_node_explorer(machines,jobs,initial_depth,
	                            distributed_active_set[idx], global_ub); 
	                    }
	                }//guided
	                otherwise{
	                    writeln("\n\n ###### error ######\n\n ###### error ######\n\n ###### error ###### \n\n ###### WRONG PARAMETERS ###### ");
	                }
	            }//scheduler
	        }//global

	        when "mixed"{

	            writeln("\n### Array of upper bounds and global on locale 0. ###\n");

	            select scheduler{
	                when "static" {
	                    forall n in distributed_active_set with (+ reduce metrics) do {
	                       metrics += fsp_johnson_mlocale_array_node_explorer(machines,jobs,initial_depth,
	                           n, global_ub, set_of_atomics);
	                    }
	                }//static
	                when "dynamic" {
	                    forall idx in distributedDynamic(c=Space, chunkSize=chunk) with (+ reduce metrics) do {
	                        metrics += fsp_johnson_mlocale_array_node_explorer(machines,jobs,initial_depth,
	                            distributed_active_set[idx], global_ub, set_of_atomics);
	                    }
	                }//dynamic
	                when "guided" {        
	                    forall idx in distributedGuided(c=Space) with (+ reduce metrics) do {
	                        metrics += fsp_johnson_mlocale_array_node_explorer(machines,jobs,initial_depth,
	                            distributed_active_set[idx], global_ub, set_of_atomics);
	                    }
	                }//guided
	                otherwise{
	                    writeln("\n\n ###### error ######\n\n ###### error ######\n\n ###### error ###### \n\n ###### WRONG PARAMETERS ###### ");
	                }
	            }//scheduler
	        }//mixed

	        when "none"{

	            var ub: c_int = global_ub.read();
	            writeln("\n ### No atomics ###\n");

	            select scheduler{
	                when "static" {
	                    forall n in distributed_active_set with (+ reduce metrics) do {
	                       metrics += fsp_johnson_mlocale_node_explorer(machines,jobs,initial_depth,
	                           n,ub);
	                    }

	                }
	                when "dynamic" {
	                    forall idx in distributedDynamic(c=Space, chunkSize=chunk) with (+ reduce metrics) do {
	                        metrics += fsp_johnson_mlocale_node_explorer(machines,jobs,initial_depth,
	                            distributed_active_set[idx], ub);
	                    }
	                }
	                when "guided" {        
	                    forall idx in distributedGuided(c=Space) with (+ reduce metrics) do {
	                        metrics += fsp_johnson_mlocale_node_explorer(machines,jobs,initial_depth,
	                            distributed_active_set[idx], ub);
	                    }
	                }
	                otherwise{
	                    writeln("\n\n ###### error ######\n\n ###### error ######\n\n ###### error ###### \n\n ###### WRONG PARAMETERS ###### ");
	                }
	                
	           }//node
	        }//no atomics
	        otherwise{
	            writeln("\n\n ###### error ######\n\n ###### error ######\n\n ###### error ###### \n\n ###### WRONG PARAMETERS ###### ");
	        }
	    }
    }//call search

}//////module