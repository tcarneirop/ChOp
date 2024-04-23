

module queens_node_module{

	use queens_constants;
	use CTypes;
	//use CPtr;

  require "headers/queens_node.h";


   extern "QueenRoot" record queens_node{
       var control: c_uint;
        var board: c_array(c_char, 12);
    };

     extern "FirstQueenRoot" record first_queens_node{
       var control: c_uint;
        var board: c_array(c_uchar, 128);
    };


}//end of module
