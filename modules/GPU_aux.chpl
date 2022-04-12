module GPU_aux{

	use CTypes;

    require "headers/GPU_aux.h";

    extern proc GPU_device_count(): c_int;

}
