
use SysCTypes;

require "add.h";



proc main(){
	
	extern proc add_c(a: c_int, b: c_int) : c_int;
	extern proc sub_int(a: c_int, b: c_int) : c_int;

	writeln("add c", add_c(1,1));

	writeln("sub int", sub_int(1,1));

}

