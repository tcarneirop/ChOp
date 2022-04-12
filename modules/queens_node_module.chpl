

module queens_node_module{

	use queens_constants;
	use CTypes;
	//use CPtr;

  require "headers/GPU_queens.h";


   extern "QueenRoot" record queens_node{
       var control: c_uint;
        var board: c_array(c_char, 12);
    };


}//end of module
