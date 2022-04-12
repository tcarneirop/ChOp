module fsp_johnson_chpl_c_headers{
	//use CPtr;
	use CTypes;
	require "headers/johnson_bound.h";

	require "headers/aux.h";

  	extern proc remplirTempsArriverDepart(minTempsArr : c_ptr(c_int), minTempsDep : c_ptr(c_int),
		machines : c_int, jobs : c_int, times : c_ptr(c_int)) : void;
	extern proc get_instance(ref machines : c_int, ref jobs : c_int, p: c_short) : c_ptr(c_int);
	extern proc print_instance(machines : c_int, jobs : c_int, times : c_ptr(c_int)) : void;
	extern proc start_vector(permutation : c_ptr(c_int), jobs : c_int) : void;
	extern proc swap(ref a : c_int, ref b : c_int) : void;
	extern proc print_permutation(permutation: c_ptr(c_int), jobs : c_int);


	extern const _MAX_MCHN_ : c_int;
	extern const _MAX_J_JOBS_ : c_int;

	// End of #define'd integer literals
	extern var tempsLag : c_ptr(c_int);
	extern var machine : c_ptr(c_int);
	extern var tabJohnson : c_ptr(c_int);
	extern var minTempsDep : c_ptr(c_int);
	extern var minTempsArr : c_ptr(c_int);
	extern var c_temps: c_ptr(c_int);


	extern proc johnson_remplirMachine(machines : c_int, machine : c_ptr(c_int)) : void;

	extern proc johnson_remplirLag(machines : c_int, jobs : c_int, machine : c_ptr(c_int),
		tempsLag : c_ptr(c_int), tempsJob : c_ptr(c_int)) : void;

	extern proc johnson_remplirTabJohnson(machines : c_int, jobs : c_int,
		tabJohnson : c_ptr(c_int),  tempsLag : c_ptr(c_int),  tempsJob : c_ptr(c_int)) : void;

	extern proc johnson_bornes_calculer(machines : c_int, jobs : c_int, job : c_ptr(c_int),
		permutation : c_ptr(c_int), limite1 : c_int, limite2 : c_int, minCmax : c_int, tempsMachines : c_ptr(c_int),
		tempsMachinesFin : c_ptr(c_int), minTempsArr : c_ptr(c_int), minTempsDep : c_ptr(c_int), machine : c_ptr(c_int),
		tempsLag : c_ptr(c_int), tempsJob : c_ptr(c_int)) : c_int;
}
