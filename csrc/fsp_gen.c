#define ANSI_C 0     /* 0:   K&R function style convention */
#define VERIFY 0     /* 1:   produce the verification file */ 
#define FIRMACIND 0  /* 0,1: first machine index           */ 

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "../headers/fsp_gen.h"




struct problem {
  long rand_time;      /* random seed for jobs */ 
  short num_jobs;      /* number of jobs */ 
  short num_mach;      /* number of machines */ 
};

#if VERIFY == 1

struct problem S[] = {
  {         0,  0, 0},
  { 873654221, 20, 5},
  {         0,  0, 0}};

#else /* VERIFY */ 
    
struct problem S[] = {
{         0,     0,  0},
                         /* 20 jobs  5 machines */ 
{ 873654221,    20,  5},  
{ 379008056,    20,  5}, 
{ 1866992158,   20,  5}, 
{ 216771124,    20,  5}, 
{ 495070989,    20,  5}, 
{ 402959317,    20,  5}, 
{ 1369363414,   20,  5}, 
{ 2021925980,   20,  5},
{ 573109518,    20,  5}, 
{ 88325120,     20,  5}, 
                          /* 20 jobs  10 machines */ 
{ 587595453,    20, 10},
{ 1401007982,   20, 10},
{ 873136276,    20, 10}, 
{ 268827376,    20, 10}, 
{ 1634173168,   20, 10},
{ 691823909,    20, 10}, 
{ 73807235,     20, 10}, 
{ 1273398721,   20, 10}, 
{ 2065119309,   20, 10}, 
{ 1672900551,   20, 10},
                          /* 20 jobs 20 machines */
{ 479340445,    20, 20},  
{ 268827376,    20, 20},
{ 1958948863,   20, 20},
{ 918272953,    20, 20},
{ 555010963,    20, 20},
{ 2010851491,   20, 20},
{ 1519833303,   20, 20},
{ 1748670931,   20, 20},
{ 1923497586,   20, 20},
{ 1829909967,   20, 20},
                          /* 50 jobs  5 machines */  
{ 1328042058,   50,  5}, 
{ 200382020,    50,  5},
{ 496319842,    50,  5},
{ 1203030903,   50,  5},
{ 1730708564,   50,  5},
{ 450926852,    50,  5},
{ 1303135678,   50,  5},
{ 1273398721,   50,  5},
{ 587288402,    50,  5},
{ 248421594,    50,  5},
                          /* 50 Jobs 10 machines */ 
{ 1958948863,   50, 10},
{ 575633267,    50, 10},
{ 655816003,    50, 10}, 
{ 1977864101,   50, 10},
{ 93805469,     50, 10},
{ 1803345551,   50, 10},  
{ 49612559,     50, 10},
{ 1899802599,   50, 10},
{ 2013025619,   50, 10},
{ 578962478,    50, 10},
                          /* 50 jobs 20 machines */ 
{ 1539989115,   50, 20},
{ 691823909,    50, 20},
{ 655816003,    50, 20}, 
{ 1315102446,   50, 20}, 
{ 1949668355,   50, 20},
{ 1923497586,   50, 20},
{ 1805594913,   50, 20},
{ 1861070898,   50, 20}, 
{ 715643788,    50, 20}, 
{ 464843328,    50, 20}, 
                          /* 100 jobs  5 machines */ 
{ 896678084,   100,  5},
{ 1179439976,  100,  5}, 
{ 1122278347,  100,  5}, 
{ 416756875,   100,  5},
{ 267829958,   100,  5}, 
{ 1835213917,  100,  5}, 
{ 1328833962,  100,  5}, 
{ 1418570761,  100,  5}, 
{ 161033112,   100,  5},
{ 304212574,   100,  5}, 
                          /* 100 jobs 10 machines */ 
{ 1539989115,  100, 10},
{ 655816003,   100, 10}, 
{ 960914243,   100, 10}, 
{ 1915696806,  100, 10},
{ 2013025619,  100, 10}, 
{ 1168140026,  100, 10}, 
{ 1923497586,  100, 10}, 
{ 167698528,   100, 10}, 
{ 1528387973,  100, 10}, 
{ 993794175,   100, 10}, 
                          /* 100 jobs 20 machines */
{ 450926852,   100, 20},
{ 1462772409,  100, 20}, 
{ 1021685265,  100, 20}, 
{ 83696007,    100, 20}, 
{ 508154254,   100, 20}, 
{ 1861070898,  100, 20}, 
{ 26482542,    100, 20}, 
{ 444956424,   100, 20}, 
{ 2115448041,  100, 20}, 
{ 118254244,   100, 20}, 
                          /* 200 jobs 10 machines */ 
{ 471503978,   200, 10},
{ 1215892992,  200, 10}, 
{ 135346136,   200, 10}, 
{ 1602504050,  200, 10}, 
{ 160037322,   200, 10}, 
{ 551454346,   200, 10}, 
{ 519485142,   200, 10}, 
{ 383947510,   200, 10}, 
{ 1968171878,  200, 10}, 
{ 540872513,   200, 10}, 
                          /* 200 jobs 20 machines */
{ 2013025619,  200, 20},
{ 475051709,   200, 20}, 
{ 914834335,   200, 20}, 
{ 810642687,   200, 20},  
{ 1019331795,  200, 20}, 
{ 2056065863,  200, 20}, 
{ 1342855162,  200, 20}, 
{ 1325809384,  200, 20}, 
{ 1988803007,  200, 20}, 
{ 765656702,   200, 20}, 
                          /* 500 jobs 20 machines */
{ 1368624604,  500, 20},
{ 450181436,   500, 20}, 
{ 1927888393,  500, 20}, 
{ 1759567256,  500, 20}, 
{ 606425239,   500, 20}, 
{ 19268348,    500, 20}, 
{ 1298201670,  500, 20}, 
{ 2041736264,  500, 20},
{ 379756761,   500, 20},
{ 28837162,    500, 20},
{          0,    0,  0}};
#endif /* VERIFY */

/* generate a random number uniformly between low and high */


int unif (long *seed, short low, short high)
{
  static long m = 2147483647, a = 16807, b = 127773, c = 2836;
  double  value_0_1;              

  long k = *seed / b;
  *seed = a * (*seed % b) - k * c;
  if(*seed < 0) *seed = *seed + m;
  value_0_1 =  *seed / (double) m;

  return (short) (low + floor(value_0_1 * (high - low + 1)));
}

/* Maximal 500 jobs and 20 machines are provided. */
/* For larger problems extend array sizes.        */ 

short d[21][501];                       /* duration */ 

/* Maximal 500 jobs and 20 machines are provided. */
/* For larger problems extend array sizes.        */ 


void generate_flow_shop(short p, int *times, int *machines, int *jobs)          /* Fill d and M according to S[p] */ 
{
  short i, j;
  long time_seed = S[p].rand_time;

  (*machines) = S[p].num_mach;
  (*jobs) = S[p].num_jobs;

  for(i = 0; i < S[p].num_mach; ++i)      /* determine a random duration */ 
    for (j = 0; j < S[p].num_jobs; ++j)   /* for all operations */ 
      d[i][j] = unif(&time_seed, 1, 99);  /* 99 = max. duration of op. */
       
   write_problem(p,times);

}

void write_problem(short p,int *times)  /* write out problem */ 
{
  short i, j;

  //printf("\nInstance Number: %hu\n", p);
  // printf("\nMachnes: %d\n", S[p].num_mach);
  // printf("\nJobs: %d\n", S[p].num_jobs);

  /* file name construction */ 
 
  for(i = 0; i < S[p].num_mach; ++i) {
    for(j = 0; j < S[p].num_jobs; ++j) {
        times[i*S[p].num_jobs+j] = d[i][j];   /* write machine and job */ 
      }
  }

  // printf("\nTimes\n");
  // for(i = 0; i < S[p].num_mach; ++i) {
  //   for(j = 0; j < S[p].num_jobs; ++j) {
      
  //        printf("%2d ", times[i*S[p].num_jobs+j]);   /* write machine and job */ 
  //     }
  //     printf("\n");                         /* newline == End of job */ 
  // }

}
