#ifndef __JOHNSON_BOUND_H_
#define __JOHNSON_BOUND_H_


#define _MAX_MCHN_ 20
#define _MAX_J_JOBS_ 30
#define _MAX_SOMME_ ((_MAX_MCHN_)*(_MAX_MCHN_-1)/2)

#define _somme_  (((machines)*(machines-1))/2)

int tempsLag[_MAX_SOMME_*_MAX_J_JOBS_];
int machine[2*_MAX_SOMME_];
int tabJohnson[_MAX_SOMME_*_MAX_J_JOBS_];

int c_temps[_MAX_MCHN_*_MAX_J_JOBS_];

int minTempsDep[_MAX_MCHN_];//read only
int minTempsArr[_MAX_MCHN_];//read only -- fill once and fire

//not in the heap -- initi and then read only
// int tempsLag[_MAX_SOMME_*_MAX_J_JOBS_];
// int machine[2*_MAX_SOMME_];
// int tabJohnson[_MAX_SOMME_*_MAX_J_JOBS_];

// int minTempsDep[_MAX_MCHN_];//read only
// int minTempsArr[_MAX_MCHN_];//read only -- fill once and fire

// //temp ...
// //int job[jobs];
// // int *pluspetit[2];
// // int tempsMachinesFin[machines]; //front
// // int tempsMachines[machines]; //back

// //int machinePairOrder[];
// //int countMachinePairs[];
 
// int nbAffectFin;
// int nbAffectDebut;
// int nbAffect;
// int borneInfCmax; 


// void johnson_set_nombres(const int jobs, const int limite1, const int limite2);

int johnson_estInf(int i, int j,int *pluspetit[2]);

int johnson_estSup(const int i, const int j,int *pluspetit[2]);

int johnson_partionner(int * ordo, int deb, int fin, int *pluspetit[2]);

void johnson_quicksort(int * ordo, int deb, int fin, int *pluspetit[2]);


void johnson_bound(const int machines, const int jobs, int * ordo, int m1, int m2, int s, 
    int *tempsLag, const int *tempsJob);

// void johnson_initCmax(const int machines, int *tmp, int *ma, int ind, int *machine, 
//     int *tempsMachines, int *minTempsArr);
void johnson_initCmax(const int machines, int *tmp, int *ma, int ind, int *machine, 
    int *tempsMachines, int *minTempsArr, const int limite1);



void johnson_cmaxFin(int * tmp, int * ma, int *tempsMachinesFin);

void johnson_set_job_jobFin(const int jobs, const int permutation[], const int limite1, const int limite2, int *job);

void johnson_heuristiqueCmax(const int jobs, int * tmp, int * ma, int ind, 
    int *tabJohnson, int *tempsLag, int *job, const int *tempsJob);

//NEW FOR MCORE
// int johnson_borneInfMakespan(const int machines, const int jobs, int *job, int *valBorneInf, int minCmax, 
//     int *minTempsArr, int *minTempsDep, int *machine, int *tempsMachines,int *tempsMachinesFin, 
//     int *tempsLag, const int *tempsJob);

int johnson_borneInfMakespan(const int machines, const int jobs, int *job, int *valBorneInf, int minCmax, 
    int *minTempsArr, int *minTempsDep, int *machine, int *tempsMachines,int *tempsMachinesFin, 
    int *tempsLag, const int limite1, const int limite2, const int *tempsJob);

void johnson_remplirLag(const int machines, const int jobs, int *machine, int *tempsLag, const int *tempsJob);

void johnson_remplirMachine(const int machines, int *machine );

void johnson_remplirTabJohnson(const int machines, const int jobs,  int *tabJohnson, 
    int *tempsLag, const int *tempsJob );

// int johnson_calculBorne(const int machines, const int jobs, int *job, int minCmax,
//     int *minTempsArr, int *minTempsDep, int *machine, int *tempsMachines,int *tempsMachinesFin, 
//     int *tempsLag, const int *tempsJob);

int johnson_calculBorne(const int machines, const int jobs, int *job, int minCmax,
    int *minTempsArr, int *minTempsDep, int *machine, int *tempsMachines,int *tempsMachinesFin, 
    int *tempsLag, const int limite1, const int limite2, const int *tempsJob);

void johnson_set_tempsMachinesFin_tempsJobFin(const int machines, const int jobs, const int permutation[], 
    int *tempsMachinesFin, const int limite2, const int *tempsJob);

void johnson_set_tempsMachines_retardDebut(const int machines, const int jobs, const int permutation[], int *tempsMachines, 
    const int limite1, const int limite2, const int *tempsJob);


int johnson_evalSolution(const int machines, const int jobs, const int *permutation, const int *tempsJob);

int johnson_bornes_calculer(const int machines, const int jobs, int *job, const int permutation[], 
    const int limite1, const int limite2,  int minCmax,int *tempsMachines, int *tempsMachinesFin,
    int *minTempsArr, int *minTempsDep, int *machine, int *tempsLag, const int* tempsJob);

void johnson_bound_search(const int machines, const int jobs, const int *tempsJob);

void johnson_call_search(short p);


#endif