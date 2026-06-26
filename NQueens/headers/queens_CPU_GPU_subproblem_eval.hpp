#ifndef QUEENS_SUB_EVAL
#define QUEENS_SUB_EVAL

#if defined(__CUDACC__) || defined(__HIPCC__)
    #define CHOP_HD __host__ __device__
#else
    #define CHOP_HD
#endif

CHOP_HD inline bool queens_is_legal_placement(const int8_t *__restrict__  board, const int r){

	bool safe = true;
	int i, rev_i, offset;
	const char base = board[r];
	// Check vertical

	for ( i = 0, rev_i = r-1, offset=1; i < r; ++i, --rev_i, offset++)
		safe &= !((board[i] == base) | ( (board[rev_i] == base-offset) |(board[rev_i] == base+offset)));
	return safe;
}

#endif