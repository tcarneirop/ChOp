#ifndef _FSP_GEN__
#define _FSP_GEN__

#define ANSI_C 0     /* 0:   K&R function style convention */
#define VERIFY 0     /* 1:   produce the verification file */ 
#define FIRMACIND 0  /* 0,1: first machine index           */ 

#include <stdio.h>
#include <math.h>


/* generate a random number uniformly between low and high */

int unif (long *seed, short low, short high);

/* Maximal 500 jobs and 20 machines are provided. */
/* For larger problems extend array sizes.        */ 

void generate_flow_shop(short p, int *times,int *machines, int *jobs); 

void write_problem(short p, int *times);


#endif