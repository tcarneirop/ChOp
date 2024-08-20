
module bitset_subproblem_explorer{

    use bitset_partial_search;
    use bitset_subproblem_module;
    use CTypes;

    proc queens_bitset_final_search(const board_size:int, const initial_depth: int, 
        ref subproblem: Bitqueens_subproblem){


    	
    	var aQueenBitRes: [0..#MAX_BOARDSIZE] int;     // results 
        var aQueenBitCol: [0..#MAX_BOARDSIZE] int;     // marks colummns which already have queens 
        var aQueenBitPosDiag: [0..#MAX_BOARDSIZE] int; // marks "positive diagonals" which already have queens 
        var aQueenBitNegDiag: [0..#MAX_BOARDSIZE] int; // marks "negative diagonals" which already have queens 
        var aStack: [0..#MAX_BOARDSIZE] int;        // we use a stack instead of recursion 


        var board_minus: int = board_size - 1;
        var mask = (1 << board_size) - 1;

        var local_num_sols: uint(64) = 0;
        var tree_size: uint(64) = 0;
        var numsolutions: uint(64) = 0;

        var stack_position: int = 0;
        
        var lsb: uint(64);
        var bitfield: uint(64);

      
        var numrows: int = initial_depth;

        var i: int;
        var odd: int = board_size & 1;


        aStack[0] = -1; // set sentinel -- signifies end of stack */


        aQueenBitRes[numrows] = subproblem.aQueenBitRes; 
        aQueenBitCol[numrows] = subproblem.aQueenBitCol; 
        aQueenBitPosDiag[numrows] = subproblem.aQueenBitPosDiag; 
        aQueenBitNegDiag[numrows] = subproblem.aQueenBitNegDiag; 
    
    
        //pnStack = aStack + pnStackPos; /* stack pointer */
        /// pnStack = aStack; /* stack pointer */
        
        bitfield = (mask & ~(aQueenBitCol[numrows] | aQueenBitNegDiag[numrows] | aQueenBitPosDiag[numrows])):uint(64);
                    
        while(true){
        	
            lsb = -(bitfield:int) & bitfield; /* this assumes a 2's complement architecture */
            
            if (0:uint(64) == bitfield)
            {
                
                if (numrows <= initial_depth) { // if sentinel hit.... */
                    break ;
                }

                stack_position-=1;
                bitfield = (aStack[stack_position]):uint(64); 
                numrows-=1;

                continue;
            }

            bitfield &= ~lsb; // toggle off this bit so we don't try it again */

            aQueenBitRes[numrows] = lsb:int; // save the result */
           
            if (numrows < board_minus) // we still have more rows to process? */
            {
            	//long long int n = numrows++; 
                var n: int = numrows;
                numrows+=1;

                aQueenBitCol[numrows] = (aQueenBitCol[n] | lsb):int;
                aQueenBitNegDiag[numrows] = ((aQueenBitNegDiag[n] | lsb) >> 1):int;
                aQueenBitPosDiag[numrows] = ((aQueenBitPosDiag[n] | lsb) << 1):int;
                 
                aStack[stack_position] = bitfield:int;
                stack_position+=1;

                bitfield = (mask & ~(aQueenBitCol[numrows] | aQueenBitNegDiag[numrows] | aQueenBitPosDiag[numrows])):uint(64);
                tree_size+=1;
                continue;
            }
            else
            {
                numsolutions+=1;
                stack_position-=1;
                bitfield = (aStack[stack_position]):uint(64);
                numrows-=1;
                continue;
            }

        }//while
    
 
       
        return (tree_size, numsolutions);
    }//final search

}//module