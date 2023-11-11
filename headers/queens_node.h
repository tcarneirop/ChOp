#ifndef QUEENS_NODE_H
#define QUEENS_NODE_H

#ifndef _EMPTY_
#define _EMPTY_ -1
#endif

typedef struct queen_root{
	unsigned int control;
	int8_t board[12];
} QueenRoot;

#endif
