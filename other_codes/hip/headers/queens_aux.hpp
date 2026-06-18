
#ifndef QUEENS_AUX_HPP
#define QUEENS_AUX_HPP


double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(hipError_t code, const char *file, int line, bool abort=true)
{
	if (code != hipSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

inline void prefixesHandleSol(QueenRoot *root_prefixes, unsigned int flag, const char *board, const int initialDepth, const int num_sol)
{
	root_prefixes[num_sol].control = flag;
	for(int i = 0; i<initialDepth;++i)
		root_prefixes[num_sol].board[i] = board[i];
}


#endif