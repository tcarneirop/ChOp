#ifndef GPU_QUEENS_H
#define GPU_QUEENS_H


typedef struct bitset_subproblem{
    
    long long  aQueenBitRes; /* results */
    long long  aQueenBitCol; /* marks columns which already have queens */
    long long  aQueenBitPosDiag; /* marks "positive diagonals" which already have queens */
    long long  aQueenBitNegDiag; /* marks "negative diagonals" which already have queens */

} Bitset_subproblem;



#endif
