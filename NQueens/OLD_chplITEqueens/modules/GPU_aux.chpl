module GPU_aux{
	
	use SysCTypes;

    require "headers/GPU_aux.h";

    extern proc GPU_device_count(): c_int;

}