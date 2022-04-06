



module queens_serial{

	use queens_node_evaluation;
	use queens_constants;


	proc queens_first_serial_bitset(const size: int, const prepro: bool): (uint(64)){

    	var bit_test : uint(32) = 0;
    	var control: uint(32) = 0;
		var board: [0..MAX] int(8) = __EMPTY__;
		var depth: int(32); //needs to be int because -1 is the break condition
		var qtd_solucoes_thread: uint(64) = 0;
		var num_sols: uint(64) = 0;
		var tree_size: uint(64) = 0;
		var metrics: uint(64);
		var _ONE_: uint(32) =  1;

		var CUTOFF = 0;
		depth = 0;

		if(prepro && size>24  ){
			writeln("Preprocessing on... ");

			board[0] = 0;
			board[1] = 2;
			board[2] = 4;
			board[3] = 1;
			board[4] = 3;
			board[5] = 8;
			board[6] = 10;
			board[7] = 12;
			board[8] = 14;

			for dpth in 0..8 do{
				bit_test |= (_ONE_<<board[dpth]);
			}

			depth=9;
			CUTOFF = 8;
			//1, 3, 5, 2, 4, 9, 11, 13, 15,
		}

		while(true){


			board[depth] = board[depth]+1;
			bit_test = 0;
        	bit_test |= (_ONE_<<board[depth]);


			if board[depth] == size then
				board[depth] = __EMPTY__;
			else{
				if (stillLegal(board, depth) && !(control &  bit_test )) {

					control |= (_ONE_<<board[depth]);
					depth +=1;
					tree_size+=1;

					if depth == size then{
						for i in 0..#size do{
							write(board[i]+1 ," - " );
						}
						writeln("\n", tree_size);
						halt();
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


			if (depth < CUTOFF) then
				break;

		}

	    metrics = tree_size;

		return metrics;
	}

	proc queens_serial_bitset(const size: uint(16)): (uint(64),uint(64)){

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

		while(true){


			board[depth] = board[depth]+1;
			bit_test = 0;
        	bit_test |= (_ONE_<<board[depth]);


			if board[depth] == size then
				board[depth] = __EMPTY__;
			else{
				if (stillLegal(board, depth) && !(control &  bit_test )) {

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

	    metrics[0] = num_sols;
	    metrics[1] = tree_size;

		return metrics;
	}

}
