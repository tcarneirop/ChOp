
use SysCTypes;

require "headers/vector_add.h";

config const size: c_int = 10:c_int;



proc print_locales_information(){
    writeln("Number of locales: ",numLocales,".");
    for loc in Locales do{
        on loc do{
            writeln("\tLocale ", here.id, ", name: ", here.name,".");
        }
    }//end for
}//print locales

proc print_results(ref vec1: c_int, ref vec2: c_int, ref vec_out: c_int){

	writeln("\n\nLocale ", here.id, " : \n");
	writeln("Inputs: ");
	writeln(vec1);
	writeln(vec2);
	writeln("Sum: ");
	writeln(vec_out);

}

proc main(){

	
	extern proc call_cuda_vector_add(): void;

	extern proc call_cuda_param_vector_add(h_a: c_ptr(c_int), h_b: c_ptr(c_int), 
		h_out: c_ptr(c_int), size:c_int ): void;


	writeln("THIS IS A CHPL CODE \n THIS IS A CHPL CODE \n THIS IS A CHPL CODE \n ");

	
	
	print_locales_information();
	writeln("\n\n");

	coforall loc in Locales do{
		on loc do {

			if (here.id > 0){
				var vec1: [0..#size] c_int = here.id:c_int;
				var vec2: [0..#size] c_int = (here.id*2):c_int;
				var vec_out: [0..#size] c_int = 0:c_int;
					
				call_cuda_param_vector_add(c_ptrTo(vec1),c_ptrTo(vec2),c_ptrTo(vec_out),size);
				writeln("\n\nLocale ", here.id, " : \n");
				writeln("Inputs: ");
				writeln(vec1);
				writeln(vec2);
				writeln("Sum: ");
				writeln(vec_out);

			}//endif
			else{
				writeln("\n\nLocale ", here.id, " : \n");
				writeln("Calling a different CUDA function: ");
				call_cuda_vector_add();
			}

		}//end on loc
	}//end coforall
	
}

