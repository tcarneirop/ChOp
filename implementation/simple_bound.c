#include "../headers/simple_bound.h"
#include "../headers/aux.h"
#include <stdio.h>
#include <limits.h>
#include <stdlib.h>


int evalsolution(const int permutation[],const int machines, const int jobs, 
    const int *times){

    //lets change for malloc
    //int temp= new int[jobs];

    int temp[jobs];

 	for(int mm=0;mm<machines;mm++) temp[mm]=0;

 	for(int j=0;j<jobs;j++)
   	{
   		int job=permutation[j];
   		//temp[0]=temp[0]+times[0][job]; ///m
        temp[0]=temp[0]+times[0*jobs + job]; ///m
         
   		for(int m=1;m<machines;m++){
   			//temp[m]=std::max(temp[m],temp[m-1])+times[m][job]; ///m
            temp[m]=max(temp[m],temp[m-1])+times[m*jobs+job];
        }
   	}

    //delete[] temp;

   	return temp[machines-1];
}


void scheduleBack(int *permut, int limit2, const int machines, const int jobs,
     int *minTempsDep, int* back, const int *times){


    int job;

    if(limit2==jobs){
        for(int i=0;i<machines;i++)
            back[i]=minTempsDep[i];//minArrive[i];
        return;
    }

    for (int j = machines - 1; j >= 0; j--)
        back[j]=0;

    //reverse schedule (allowing incremental update)
    for (int k = jobs - 1; k>=limit2; k--) {
        job=permut[k];
        //back[machines-1]+= times[machines-1][job];//ptm[(_machines_-1) * _jobs_ + job];
        back[machines-1]+= times[(machines-1)*jobs+job]; ///m

        for (int j = machines - 2; j >= 0; j--){
            //back[j]=std::max(back[j],back[j+1])+times[j][job]; ///m
            back[j]=max(back[j],back[j+1])+times[j*jobs+job]; ///m
        
        }
    }
}



void scheduleFront(int *permut, int limit1,int limit2, 
    const int machines, const int jobs, int *minTempsArr, 
    int *front, const int *times){


    int job;

    if(limit1==-1){
        for(int i=0;i<machines;i++)
            front[i]=minTempsArr[i];//minRelease[i];
        return;
    }

    for(int i=0;i<machines;i++)
        front[i]=0;

    for(int i=0;i<limit1+1;i++){
        job=permut[i];
        //front[0]+=times[0][job];///m
        front[0]+=times[0*jobs+job];///m
        
        for(int j=1;j<machines;j++){
            //front[j]=std::max(front[j-1],front[j])+times[j][job]; ///m
            front[j]=max(front[j-1],front[j])+times[j*jobs+job];
        }
    }

    for(int j=1;j<machines;j++){
        front[j]=max(front[j-1],front[j]);
    }
}




void sumUnscheduled(const int *permut, int limit1, int limit2, 
    const int machines, const int jobs, int *remain, const int *times){


    int job;

    for (int j = 0; j < machines; j++)remain[j]=0;

    for (int k = limit1+1; k<limit2; k++) {
        job=permut[k];
        for (int j = 0; j < machines; j++){
            //remain[j]+= times[j][job];///m
            remain[j]+= times[j*jobs+job];///m
        }
    }
}



int simple_bornes_calculer(int permutation[], int limite1, int limite2, 
    const int machines, const int jobs, int *remain, int *front, 
     int *back, int *minTempsArr, int *minTempsDep, const int *times){



    //scheduleFront(permutation, limite1, limite2);

    scheduleFront(permutation, limite1, limite2, machines, jobs, minTempsArr, front, times);
    scheduleBack(permutation, limite2, machines, jobs, minTempsDep, back, times);
    sumUnscheduled(permutation, limite1, limite2, machines, jobs, remain, times);

    //sumUnscheduled(permutation, limite1, limite2);
    //(const int *permut, int limit1, int limit2, 
    //const int machines, const int jobs, int *remain, const int *times)

    int lb;
    int tmp0,tmp1,cmax;

    tmp0=front[0]+remain[0];
    lb=tmp0+back[0];

    for(int j=1;j<machines;j++){
        tmp1=front[j]+remain[j];
        tmp1=max(tmp1,tmp0);
        cmax=tmp1+back[j];
//        printf("%d\n",cmax);
        lb=max(lb,cmax);
        tmp0=tmp1;
    }

    return lb;
}




void simple_bound_search(int machines, int jobs, int *times){


    register unsigned int flag = 0;
    register int bit_test = 0;
    register int i, depth; //para dizer que 0-1 ja foi visitado e a busca comeca de 1, bote 2
    register unsigned long long int local_tree = 0ULL;
    int num_sol = 0;
    int lowerbound = 0;
    int incumbent = 1713;
    //int incumbent = 1713;
    register int p1;

    int minTempsDep[machines];//read only
    int minTempsArr[machines];//read only -- fill once and fire

    int front[machines];//private could be inside the heap -- insithe the bounding func
    int back[machines];//private
    int remain[machines];//private

    int permutation[jobs];//
   
    int scheduled[jobs];//flag
    int position[jobs];//position -- swap stuff:

 
    remplirTempsArriverDepart(minTempsArr, minTempsDep, machines,jobs,times);

    start_vector(position,jobs);
    start_vector(permutation,jobs);


    for (i = 0; i < jobs; ++i) { //
        scheduled[i] = -1;
    }

    depth = 0;

    do{
        scheduled[depth]++;
        bit_test = 0;
        bit_test |= (1<<scheduled[depth]);

        if(scheduled[depth] == jobs){
            scheduled[depth] = -1;
        }else{
                //se colocar aqui serao invalidas
                if (!(flag & bit_test)){ //is valid

                    p1 = permutation[depth];
                    swap(&permutation[depth],&permutation[position[scheduled[depth]]]);
                    swap(&position[scheduled[depth]],&position[p1]);

                    lowerbound = simple_bornes_calculer(permutation, depth, jobs,
                         machines, jobs, remain, front, back, 
                         minTempsArr, minTempsDep, times);
                    
                    if(lowerbound<incumbent){//is it feasible

                        flag |= (1ULL<<scheduled[depth]);
                        depth++;
                        local_tree++;
                        //print_subsol(scheduled, depth);
                        //print_permutation(permutation,_jobs_);
                        if (depth == jobs){ //handle solution
                            if(lowerbound < incumbent){
                                num_sol++;
                                incumbent = lowerbound;
                                printf("\n");
                                printf("\nnew incumbent solution: %d.\n",lowerbound);
                                print_permutation(permutation,jobs);
                            }
                        }else continue;// not a complete one
                    }else continue; //not feasible
                }else continue; //not valid
        }//else

        depth--;
        flag &= ~(1ULL<<scheduled[depth]);

    }while(depth >= 0);

    printf("\nNumber of solutions found: %d.\n", num_sol );
    printf("\nTree size: %llu.\n", local_tree);
    printf("\n\tBest solution: %d.\n", incumbent);
}

void simple_bound_call_search(short p){

    int jobs;
    int machines;
    int *times = get_instance(&machines,&jobs,p);
    print_instance(machines, jobs, times);
    simple_bound_search(machines, jobs, times);

}///////////////
