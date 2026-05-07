



module queens_serial{

	use queens_node_evaluation;
	use queens_constants;

	proc queens_serial_search(const size: uint(16)): (uint(64),uint(64)){

    	var bit_test : uint(32) = 0;
    	var control: uint(32) = 0;
		var board: [0..MAX] int(8) = __EMPTY__;
		var depth: int(32); //needs to be int because -1 is the break condition
		var qtd_solucoes_thread: uint(64) = 0;
		var num_sols: uint(64) = 0;
		var tree_size: uint(64) = 0;
		var metrics: (uint(64),uint(64));
		var _ONE_: uint(32) =  1;
		depth = 0;

		var break_cond: uint(64) =  (size/2):uint(64) + (size:uint(64) & 1:uint(64));
		

		//condition to avoid mirrored sols
		if(avoidMirrored) then writeln("\n\t##### Avoiding Mirrored Solutions #####\n");
			

		while(true){


			board[depth] = board[depth]+1;
			bit_test = 0;
        	bit_test |= (_ONE_<<board[depth]);


			if board[depth] == size then
				board[depth] = __EMPTY__;
			else{
				if (stillLegal(board, depth) && !(control &  bit_test )) {


					if(avoidMirrored) then{
						//this is enough to avoid reflections
						if(depth == 1){

							if(size:uint(64) & 1:uint(64)){
								if board[0] == break_cond-1 && board[1] > board[0] then break;
							}
							else{
								if board[0] == break_cond then break;
							}
						}
					}//mirrored 
						

					control |= (_ONE_<<board[depth]);
					depth +=1;
					tree_size+=1;

					if depth == size then{
						num_sols+=1;

					}
					else
						continue;
				}
				else
					continue;
			}

			depth -= 1;
			control &= ~(_ONE_<<board[depth]);


			if (depth < 0) then
				break;

		}

	    metrics[0] = num_sols * 2 *(avoidMirrored:uint(64));
	    metrics[1] = tree_size;

		return metrics;
	}

}
