#ifndef __SIMPLE_BOUND__
#define __SIMPLE_BOUND__


// void remplirTempsArriverDepart(int *minTempsArr, int *minTempsDep, 
//     const int machines, const int jobs, const int * times);


#define _MAX_S_MCHN_ 20
#define _MAX_S_JOBS_ 30

int c_temps[_MAX_S_MCHN_*_MAX_S_JOBS_];
int minTempsDep[_MAX_S_MCHN_];//read only
int minTempsArr[_MAX_S_MCHN_];//read only -- fill once and fire


int evalsolution(const int permutation[],const int machines, const int jobs, 
    const int *times);

void scheduleBack(int *permut, int limit2, const int machines, const int jobs,
     int *minTempsDep, int* back, const int *times);

void scheduleFront(int *permut, int limit1,int limit2, 
    const int machines, const int jobs, int *minTempsArr, 
    int *front, const int *times);

void sumUnscheduled(const int *permut, int limit1, int limit2, 
    const int machines, const int jobs, int *remain, const int *times);

int simple_bornes_calculer(int permutation[], int limite1, int limite2, 
    const int machines, const int jobs, int *remain, int *front, 
     int *back, int *minTempsArr, int *minTempsDep, const int *times);

// int* get_instance(int *machines, int *jobs);

// void print_instance(int machines, int jobs, int *times);

void simple_bound_search(int machines, int jobs, int *times);

void simple_bound_call_search(short p);


#endif