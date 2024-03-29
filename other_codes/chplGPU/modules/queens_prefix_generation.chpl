
module queens_prefix_generation{

	use queens_node_evaluation;
	use queens_constants;
	use queens_node_module;
	use CTypes;


	proc queens_get_number_prefixes(const size: uint(16), const initial_depth: int(32)): uint(64){

		 var number: uint(64) = size;

		 if initial_depth == 1 then
			return number;

		 for d in 2..initial_depth do
			number *= (((size-d)+1):uint(64));

		 return number;

 	}//get maximum number of prefixes



	proc queens_node_generate_initial_prefixes(const size: uint(16), const initial_depth: int(32),
		ref set_of_nodes: [] queens_node): (uint(64),uint(64)){


		var bit_test : uint(32) = 0;
		var control: uint(32) = 0;
		var board: [0..MAX] int(8) = __EMPTY__;
		var depth: int(32); //needs to be int because -1 is the break condition
		var qtd_solucoes_thread: uint(64) = 0;
		var num_prefixes: uint(64) = 0;
		var tree_size: uint(64) = 0;
		/* var metrics: (uint(64),uint(64)); */
		var position: uint(64) = 0;
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

						if depth == initial_depth then{
							//here we generate the backtracking roots
							set_of_nodes[num_prefixes].control = control;
							//position = num_prefixes* (initial_depth:uint(64));
							for i in 0..initial_depth-1 do
								set_of_nodes[num_prefixes].board[i] = board[i];

							num_prefixes+=1;
						}
						else
							continue;
					}
					else
						continue;
			}//else

			depth -= 1;
			control &= ~(_ONE_<<board[depth]);

			if (depth < 0) then
					break;
		}//while true

		return(num_prefixes,tree_size);
	}


}//module
