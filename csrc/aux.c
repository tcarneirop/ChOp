#include "../headers/aux.h"
#include "../headers/fsp_gen.h"
#include <limits.h>
#include <stdlib.h>
#include <stdio.h>



void remplirTempsArriverDepart(int* minTempsArr, int* minTempsDep, 
    const int machines, const int jobs, const int* times){
    
    int t0,tmin;

    minTempsArr[0]=0;

    //for machines
    for(int m=1;m<machines;m++){
        tmin=INT_MAX;
        //find smallest date a job can start on m
        for(int j=0;j<jobs;j++){
            t0=0;
            for(int mm=0;mm<m;mm++){
                //t0+=times[mm][j]; ///m
                t0+=times[mm*jobs+j];
            }
            if(t0<tmin)tmin=t0;
        }
        minTempsArr[m]=tmin;
    }

    minTempsDep[machines-1]=0;
    //for machines
    for(int m=machines-2;m>=0;m--){
        tmin=INT_MAX;
        //find smallest date a job can start on m
        for(int j=0;j<jobs;j++){
            t0=0;
            for(int mm=machines-1;mm>m;mm--){
                //t0+=times[mm][j]; ///m
                t0+=times[mm*jobs+j];
            }
            if(t0<tmin)tmin=t0;
        }
//        printf("%d\n",tmin);
        minTempsDep[m]=tmin;
    }
}



void print_subsol(int *permutation, int depth){
    printf("\nSubsolution: \n" );

    for(int i = 0; i<depth;++i){
        printf(" %d - ",permutation[i]);
    }

    printf("\n");
}



void print_permutation(int *permutation, int jobs){
    printf("\nPermutation: \n" );

    for(int i = 0; i<jobs;++i){
        printf(" %d - ",permutation[i]);
    }

    printf("\n");
}



void print_instance(int machines, int jobs, int *times){

    //scanf("%d", &upper_bound);
    printf("\nInstance (M x J): \n\n%2d x %2d\n", machines,jobs);

    for (int m = 0; m < ( machines ); m++) {
        for(int j = 0; j < jobs; ++j ){
            printf(" %2d ", times[m*jobs + j]);
        }
        printf("\n");
    }
}/////////////////


void start_vector(int *permutation, int jobs){

    for(int i = 0; i<jobs;++i){
        permutation[i] = i;
    }
}

void swap(int *a,int *b){
    int tmp=*b;
    *b=*a;
    *a=tmp;
}


int max(int a, int b){

    return (a>b) ?  a :  b;

}


//the time matrix
int* get_instance(int *machines, int *jobs, short inst_num){

    //scanf("%d", &upper_bound);
    int *instance = (int*)(malloc(sizeof(int)*5000));
    generate_flow_shop(inst_num, instance,machines,jobs);

    // print_instance( *machines,*jobs, instance);
    //write_problem(inst_num,instance); 

    // for (i = 0; i < ( m * j ); i++) {
    //     scanf("%d", &instance[i]);
    // }

    // (*machines) = m;
    // (*jobs) = j;

    return instance;
}

