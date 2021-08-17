#ifndef __AUX_H_
#define __AUX_H_



int max(int a, int b);

void start_vector(int *permutation, int jobs);

void swap(int *a,int *b);

int* get_instance(int *machines, int *jobs, short p);

void print_subsol(int *permutation, int depth);
void print_instance(int machines, int jobs, int *times);
void print_permutation(int *permutation, int jobs);

void remplirTempsArriverDepart(int *minTempsArr, int *minTempsDep, 
    const int machines, const int jobs, const int * times);



#endif