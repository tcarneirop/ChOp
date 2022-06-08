
module queens_tree_exploration{

    use queens_constants;
    use queens_node_module;
    use queens_node_evaluation;
    use CTypes;
    //use CPtr;

    proc queens_subtree_explorer(const size: uint(16),const cutoff_depth: int(32), ref node: queens_node ):(uint(64),uint(64)){

        var bit_test : uint(32) = 0;
        var control: uint(32) = 0;
        var board: [0..MAX] int(8) = __EMPTY__;
        var depth: int(32); //needs to be int because -1 is the break condition
        var number_sols: uint(64) = 0;
        var tree_size: uint(64) = 0;
        var _ONE_: uint(32) =  1;
        var cutoff64: int(64) = cutoff_depth:int(64);

        //initialization
        depth = cutoff_depth;
        control = node.control;

        for i in 0..cutoff64-1{
            board[i] = node.board[i];
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
                        number_sols+=1;

                    }
                    else
                        continue;
                }
                else
                    continue;
            }//else

            depth -= 1;
            control &= ~(_ONE_<<board[depth]);

            if (depth < cutoff_depth) then
                break;
        }//while true

        return(number_sols,tree_size);

    }//end of subtree explorer


    proc queens_node_subtree_exporer(const size: uint(16),const cutoff_depth: int(32),
         node_board: c_array(c_char,12), node_control: uint(32) ):(uint(64),uint(64)){

        //writeln("\n##################################################\n");

        var bit_test : uint(32) = 0;
        var control: uint(32) = 0;
        var board: [0..MAX] int(8) = __EMPTY__;
        var depth: int(32); //needs to be int because -1 is the break condition
        var number_sols: uint(64) = 0;
        var tree_size: uint(64) = 0;
        var _ONE_: uint(32) =  1;
        var cutoff64: int(64) = cutoff_depth:int(64);

        //initialization
        depth = cutoff_depth;
        control = node_control;

        for i in 0..cutoff64-1{
            board[i] = node_board[i];
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
                        number_sols+=1;
                    }
                    else
                        continue;
                }
                else
                    continue;
            }//else

            depth -= 1;
            control &= ~(_ONE_<<board[depth]);

            if (depth < cutoff_depth) then
                break;
        }//while true

        return(number_sols,tree_size);

    }//end of subtree explorer

}
//end of module
