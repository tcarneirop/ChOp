extern proc GPU_call_cuda_queens(size: uint(16), initial_depth:c_int, n_explorers:c_uint, 		active_set_h: c_ptr(queens_node),vector_of_tree_size_h: c_ptr(c_uint), sols_h: c_ptr(c_uint)): void;
///////////////////////////////////////////////////////////////////////////
//CUDA Vectors
///////////////////////////////////////////////////////////////////////////
var vector_of_tree_size_h: [0..#75580635] c_uint;
var sols_h: [0..#75580635] c_uint;
//var active_set_h: [0..#] queens_node;
var maximum_number_prefixes: int = 75580635;
var active_set_h: [0..#maximum_number_prefixes] queens_node;
///////////////////////////////////////////////////////////////////////////
//Single-locale wrapper
///////////////////////////////////////////////////////////////////////////
var GPUWrapper = lambda (lo:int, hi: int, n_explorers: int) {
  		GPU_call_cuda_queens(size, initial_depth, n_explorers:c_uint, c_ptrTo(active_set_h),c_ptrTo(vector_of_tree_size_h), c_ptrTo(sols_h));};
///////////////////////////////////////////////////////////////////////////
//Generating the initial Pool of nodes and metrics
///////////////////////////////////////////////////////////////////////////
metrics+= queens_node_generate_initial_prefixes(size, initial_depth, active_set_h);
n_explorers = metrics[0]:int;
initial_tree_size = metrics[1];
metrics[0] = 0; //restarting for the parallel search_type
metrics[1] = 0;
///////////////////////////////////////////////////////////////////////////
//Distributed -- generating distributed Pool
///////////////////////////////////////////////////////////////////////////
var D: domain(1) dmapped Block(boundingBox = {0..#n_explorers}) = {0..#n_explorers};
var dist_vector_of_tree_size_h: [D] c_uint;
var dist_sols_h: [D] c_uint;
var dist_active_set_h: [D] queens_node;
if distributed then {
	bulktransfer.start();
	var centralized_active_set: [0..#n_explorers] queens_node;
	forall i in 0..#n_explorers do centralized_active_set[i] = active_set_h[i];
    dist_active_set_h = centralized_active_set;
    bulktransfer.stop();
///////////////////////////////////////////////////////////////////////////
//Distributed wrapper
///////////////////////////////////////////////////////////////////////////
var DISTGPUWrapper = lambda (lo:int, hi: int, n_explorers: int) {
		ref ldist_active_set_h= dist_active_set_h.localSlice(lo .. hi);
  		ref ldist_vector_of_tree_size_h = dist_vector_of_tree_size_h.localSlice(lo .. hi);
		ref ldist_sols_h = dist_sols_h.localSlice(lo .. hi);
  		GPU_call_cuda_queens(size, initial_depth, n_explorers:c_uint, c_ptrTo(ldist_active_set_h),c_ptrTo(ldist_vector_of_tree_size_h), c_ptrTo(ldist_sols_h));};
////////////////////////////////////////////////////////////////////
//// Search itself
////////////////////////////////////////////////////////////////////
		forall i in GPU(0..#(n_explorers:int), GPUWrapper, 0){
		var dist_redTree = (+ reduce dist_vector_of_tree_size_h):uint(64);
		var dist_redSol  = (+ reduce dist_sols_h):uint(64);
		final_tree_size = dist_redTree + initial_tree_size;
		final_sol = dist_redSol;
