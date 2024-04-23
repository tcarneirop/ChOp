#ifndef QUEENS_NODE_H
#define QUEENS_NODE_H

#ifndef _EMPTY_
#define _EMPTY_ -1
#endif

typedef struct queen_root{
	unsigned int control;
	uint8_t board[12];
} QueenRoot;

typedef struct first_queen_root{
	unsigned int control;
	uint8_t board[128];
} FirstQueenRoot;

#endif
