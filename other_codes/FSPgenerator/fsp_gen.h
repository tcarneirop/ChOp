#ifndef _FSP_GEN__
#define _FSP_GEN__


/* generate a random number uniformly between low and high */

int unif (long *seed, short low, short high);

/* Maximal 500 jobs and 20 machines are provided. */
/* For larger problems extend array sizes.        */ 

                  /* duration */ 

void generate_flow_shop(short p);


void write_problem(short p);


#endif