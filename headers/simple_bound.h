#ifndef __SIMPLE_BOUND__
#define __SIMPLE_BOUND__


// void remplirTempsArriverDepart(int *minTempsArr_s, int *minTempsDep_s, 
//     const int machines, const int jobs, const int * times);


#define _MAX_S_MCHN_ 20
#define _MAX_S_JOBS_ 30

extern int c_temps[_MAX_S_MCHN_*_MAX_S_JOBS_];
extern int minTempsDep_s[_MAX_S_MCHN_];//read only
extern int minTempsArr_s[_MAX_S_MCHN_];//read only -- fill once and fire


int evalsolution(const int permutation[],const int machines, const int jobs, 
    const int *times);

void scheduleBack(int *permut, int limit2, const int machines, const int jobs,
     int *minTempsDep_s, int* back, const int *times);

void scheduleFront(int *permut, int limit1,int limit2, 
    const int machines, const int jobs, int *minTempsArr_s, 
    int *front, const int *times);

void sumUnscheduled(const int *permut, int limit1, int limit2, 
    const int machines, const int jobs, int *remain, const int *times);

int simple_bornes_calculer(int permutation[], int limite1, int limite2, 
    const int machines, const int jobs, int *remain, int *front, 
     int *back, int *minTempsArr_s, int *minTempsDep_s, const int *times);

void simple_bound_search(int machines, int jobs, int *times);

void simple_bound_call_search(short p);


#endif