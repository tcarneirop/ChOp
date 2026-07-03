#ifndef QUEENS_SUBPPROBLEM_HPP
#define QUEENS_SUBPPROBLEM_HPP


#define EMPTY        -1
#define MAX_SIZE        24
#define SUBPROBLEM_SIZE 10


/// this is used to check if the solution produced is correct
unsigned long long check_sols_number[] = {0,	0,	0,	2,	10,	4,	40,	92,	352,	724,	2680,	14200,	73712,	
365596,	2279184,	14772512,	95815104,	666090624,	4968057848,	39029188884,314666222712,2691008701644,24233937684440,
227514171973736 };


typedef __attribute__((aligned(4))) struct queen_root{
		unsigned int control;
		int8_t board[SUBPROBLEM_SIZE]; //maximum depth of the solution space.
} QueenRoot;


#endif