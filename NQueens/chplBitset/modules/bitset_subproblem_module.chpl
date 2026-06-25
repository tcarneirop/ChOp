

module bitset_subproblem_module{

     use CTypes;

     require "headers/subproblem.h";
     
     const MAX_BOARDSIZE: int = 24;

     extern "Bitset_subproblem" record Bitqueens_subproblem{
          
          var  aQueenBitRes: c_longlong; /* results */
          var  aQueenBitCol: c_longlong; /* marks columns which already have queens */
          var  aQueenBitPosDiag: c_longlong; /* marks "positive diagonals" which already have queens */
          var  aQueenBitNegDiag: c_longlong; /* marks "negative diagonals" which already have queens */
     };


}//end of module
