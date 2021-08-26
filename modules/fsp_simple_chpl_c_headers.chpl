module fsp_simple_chpl_c_headers{
	
	use SysCTypes;
	use CPtr;
	
	require "headers/simple_bound.h";
	require "headers/aux.h";


	extern const _MAX_S_MCHN_ : c_int;
	extern const _MAX_S_JOBS_ : c_int;
	extern var minTempsDep_s : c_ptr(c_int);
	extern var minTempsArr_s : c_ptr(c_int);
	extern var c_temps: c_ptr(c_int);

	
	
  	extern proc simple_bornes_calculer(permutation : c_ptr(c_int), limite1 : c_int, limite2 : c_int, machines : c_int, jobs : c_int,  remain : c_ptr(c_int), front : c_ptr(c_int), back :c_ptr(c_int),
  		minTempsArr_s : c_ptr(c_int), minTempsDep_s : c_ptr(c_int), times : c_ptr(c_int)) : c_int;


	extern proc remplirTempsArriverDepart(minTempsArr_s : c_ptr(c_int), minTempsDep_s : c_ptr(c_int),
		machines : c_int, jobs : c_int, times : c_ptr(c_int)) : void;
	extern proc get_instance(ref machines : c_int, ref jobs : c_int, p:c_short) : c_ptr(c_int);
	extern proc print_instance(machines : c_int, jobs : c_int, times : c_ptr(c_int)) : void;
	extern proc start_vector(permutation : c_ptr(c_int), jobs : c_int) : void;
	extern proc swap(ref a : c_int, ref b : c_int) : void;

}