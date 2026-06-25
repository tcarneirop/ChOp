// Generated with c2chapel version 0.1.0

// Header given to c2chapel:

require "headers/simple_bound.h";

// Note: Generated with fake std headers

// extern proc start_vector(ref permutation : c_int, jobs : c_int) : void;

// extern proc swap(ref a : c_int, ref b : c_int) : void;


//extern proc remplirTempsArriverDepart(ref minTempsArr : c_int, ref minTempsDep : c_int, machines : c_int, jobs : c_int, ref times : c_int) : void;

// extern proc evalsolution(permutation : c_ptr(c_int), machines : c_int, jobs : c_int, ref times : c_int) : c_int;

// extern proc scheduleBack(ref permut : c_int, limit2 : c_int, machines : c_int, jobs : c_int, ref minTempsDep : c_int, ref back : c_int, ref times : c_int) : void;

// extern proc scheduleFront(ref permut : c_int, limit1 : c_int, limit2 : c_int, machines : c_int, jobs : c_int, ref minTempsArr : c_int, ref front : c_int, ref times : c_int) : void;

// extern proc sumUnscheduled(ref permut : c_int, limit1 : c_int, limit2 : c_int, machines : c_int, jobs : c_int, ref remain : c_int, ref times : c_int) : void;

// extern proc simple_bornes_calculer(permutation : c_ptr(c_int), limite1 : c_int, limite2 : c_int, machines : c_int, jobs : c_int, ref remain : c_int, ref front : c_int, ref back : c_int, ref minTempsArr : c_int, ref minTempsDep : c_int, ref times : c_int) : c_int;

// extern proc get_instance(ref machines : c_int, ref jobs : c_int) : c_ptr(c_int);

// extern proc print_instance(machines : c_int, jobs : c_int, ref times : c_int) : void;

// extern proc simple_bound_search(machines : c_int, jobs : c_int, ref times : c_int) : void;

extern proc simple_bound_call_search() : void;




writeln("Hello");
writeln("alor meu consagrado");

simple_bound_call_search();