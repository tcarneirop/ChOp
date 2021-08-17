
use SysCTypes;

require "vector_add.h";



proc main(){


	
	extern proc call_cuda_vector_add(): void;

	writeln("THIS IS A CHPL CODE \n THIS IS A CHPL CODE \n THIS IS A CHPL CODE \n ");
	
	
	coforall loc in Locales do
		on loc do call_cuda_vector_add();
	
}

