#ifndef QUEENS_SUB_EVAL
#define QUEENS_SUB_EVAL

#if defined(__CUDACC__) || defined(__HIPCC__)
    #define CHOP_HD __host__ __device__
#else
    #define CHOP_HD
#endif

#if defined(_OPENMP) && defined(ENABLE_OMP_OFFLOAD)
    #pragma omp declare target
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

#if defined(_OPENMP) && defined(ENABLE_OMP_OFFLOAD)
    #pragma omp end declare target
#endif

#endif
