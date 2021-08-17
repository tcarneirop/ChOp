#include "../headers/johnson_bound.h"
#include "../headers/aux.h"
#include <stdio.h>
#include <limits.h>
#include <stdlib.h>

#define _MAX_SOMME_ ((_MAX_MCHN_)*(_MAX_MCHN_-1)/2)

#define _somme_  (((machines)*(machines-1))/2)

//not in the heap -- initi and then read only
extern int tempsLag[_MAX_SOMME_*_MAX_J_JOBS_];
extern int machine[2*_MAX_SOMME_];
extern int tabJohnson[_MAX_SOMME_*_MAX_J_JOBS_];

extern int minTempsDep[_MAX_MCHN_];//read only
extern int minTempsArr[_MAX_MCHN_];//read only -- fill once and fire


//temp ...
//int job[jobs];
// int *pluspetit[2];
// int tempsMachinesFin[machines]; //front
// int tempsMachines[machines]; //back

//int machinePairOrder[];
//int countMachinePairs[];
 

//NEW FOR MCORE
// int nbAffectFin;
// int nbAffectDebut;
// int nbAffect;


//NEW FOR MCORE
// void johnson_set_nombres(const int jobs, const int limite1, const int limite2)
// {
//     nbAffectDebut = limite1 + 1;
//     nbAffectFin   = jobs - limite2;
//     nbAffect      = nbAffectDebut + nbAffectFin;
// }


int johnson_estInf(int i, int j,int *pluspetit[2])
{
    if (pluspetit[0][i] == pluspetit[0][j]) {
        if (pluspetit[0][i] == 1)
            return pluspetit[1][i] < pluspetit[1][j];

        return pluspetit[1][i] > pluspetit[1][j];
    }
    return pluspetit[0][i] < pluspetit[0][j];
}


int johnson_estSup(const int i, const int j,int *pluspetit[2]){

    if (pluspetit[0][i] == pluspetit[0][j]) {
        if (pluspetit[0][i] == 1)
            return pluspetit[1][i] > pluspetit[1][j];

        return pluspetit[1][i] < pluspetit[1][j];
    }
    return pluspetit[0][i] > pluspetit[0][j];
}


int johnson_partionner(int * ordo, int deb, int fin, int *pluspetit[2]){

    int d = deb - 1;
    int f = fin + 1;
    int mem, pivot = ordo[deb];

    do { do f--;
         while (johnson_estSup(ordo[f], pivot, pluspetit));
         do d++;
         while (johnson_estInf(ordo[d], pivot, pluspetit));

         if (d < f) {
             mem     = ordo[d];
             ordo[d] = ordo[f];
             ordo[f] = mem;
         }
    } while (d < f);
    return f;
}


void johnson_quicksort(int * ordo, int deb, int fin, int *pluspetit[2])
{
    int k;

    if ((fin - deb) > 0) {
        k = johnson_partionner(ordo, deb, fin, pluspetit);
        johnson_quicksort(ordo, deb, k,pluspetit);
        johnson_quicksort(ordo, k + 1, fin,pluspetit);
    }
}



void johnson_bound(const int machines, const int jobs, int * ordo, int m1, int m2, int s, 
    int *tempsLag, const int *tempsJob){

    //looks like it is used for the quicksort
    int *pluspetit[2];

    pluspetit[0] = (int *) malloc((jobs) * sizeof(int));
    pluspetit[1] = (int *) malloc((jobs) * sizeof(int));
    for (int i = 0; i < jobs; i++) {
        ordo[i] = i;
        if (tempsJob[m1*jobs+i] < tempsJob[m2*jobs+i]) {
            pluspetit[0][i] = 1;
            pluspetit[1][i] = tempsJob[m1*jobs+i] + tempsLag[s*jobs+i];
        } else   { pluspetit[0][i] = 2;
                   pluspetit[1][i] = tempsJob[m2*jobs+i] + tempsLag[s*jobs+i];
        }
        //pluspetit[1] contains the smaller of JohPTM_i_m1 and JohPTM_i_m2
    }
    johnson_quicksort(ordo, 0, (jobs - 1),pluspetit);
    free(pluspetit[0]);
    free(pluspetit[1]);
}


void johnson_initCmax(const int machines, int *tmp, int *ma, int ind, int *machine, 
    int *tempsMachines, int *minTempsArr, const int limite1) {
    
    ma[0] = machine[ind];
    ma[1] = machine[_somme_+ ind];
    /*On verifie si il y a deja un job affecte au debut*/
    int m0 = ma[0];
    int m1 = ma[1];
    int nbAffectDebut = limite1 + 1;
 
    //get release dates...
    //earliest time processing can start on m0 and m1 (under partial schedeule BEGIN)
    if (nbAffectDebut != 0) {
        //mini
        //tmp[0] = m0>0?mini:noeud->tempsMachines[m0];//C(permBegin,m0)
        tmp[0] = tempsMachines[m0];//C(permBegin,m0)
        tmp[1] = tempsMachines[m1];//C(permBegin,m1)
    } else  {
        //min release on m0,m1
        tmp[0] = minTempsArr[m0];
        tmp[1] = minTempsArr[m1];
    }
}


void johnson_cmaxFin(int * tmp, int * ma, int *tempsMachinesFin)
{
    if (tmp[1] + tempsMachinesFin[ma[1]] > tmp[0]
      + tempsMachinesFin[ma[0]])
        tmp[1] = tmp[1] + tempsMachinesFin[ma[1]];
    else
        tmp[1] = tmp[0] + tempsMachinesFin[ma[0]];
}




void johnson_set_job_jobFin(const int jobs, const int permutation[], const int limite1, const int limite2, int *job){
    
    for (int j = 0; j <= limite1; j++) job[permutation[j]] = j + 1;
    for (int j = limite1 + 1; j < limite2; j++){
        job[permutation[j]] = 0;
    }
    for (int j = limite2; j < jobs; j++) {
        job[permutation[j]] = j + 1;
    }
}


void johnson_heuristiqueCmax(const int jobs, int * tmp, int * ma, int ind, 
    int *tabJohnson, int *tempsLag, int *job, const int *tempsJob){

    int jobCour;
    int tmp0 = tmp[0];
    int tmp1 = tmp[1];
    int ma0  = ma[0];
    int ma1  = ma[1];

    //int mintm1=tmp1;
    for (int j = 0; j < jobs; j++) {
        //ind=row in tabJohnson corresponding to ma0 and ma1
        //tabJohnson contains jobs sorted according to johnson's rule
        jobCour = tabJohnson[ind*jobs+j];
        //j-loop is on unscheduled jobs... (==0 if jobCour is unscheduled)
        if (job[jobCour] == 0) {
            //add jobCour to ma0 and ma1
            tmp0 += tempsJob[ma0*jobs+jobCour];
            if (tmp1 > tmp0 + tempsLag[ind*jobs+jobCour])
                tmp1 += tempsJob[ma1*jobs+jobCour];
            else
                tmp1 = tmp0 + tempsLag[ind*jobs+jobCour] + tempsJob[ma1*jobs+jobCour];
        }
    }
    tmp[0] = tmp0;
    tmp[1] = tmp1;

    //printf("%d\n",tmp1-mintm1);
}


int johnson_borneInfMakespan(const int machines, const int jobs, int *job, int *valBorneInf, int minCmax, 
    int *minTempsArr, int *minTempsDep, int *machine, int *tempsMachines,int *tempsMachinesFin, 
    int *tempsLag, const int limite1, const int limite2, const int *tempsJob){


    int moinsBon = 0;
    int ma[2]; /*Contient les rang des deux machines considere.*/
    int tmp[2]; /*Contient les temps sur les machines considere*/

    int i,j,l;
    int bestind=0;

    //NEW FOR MCORE
    int nbAffectFin = jobs - limite2;


    //sort machine-pairs
    // i = 1, j = 2;
    // while (i < _somme_) {
    //     if (countMachinePairs[machinePairOrder[i - 1]] < countMachinePairs[machinePairOrder[i]]) {
    //         myswap(&machinePairOrder[i - 1], &machinePairOrder[i]);
    //          if ((--i)) continue;
    //      }
    //      i = j++;
    // }
    //int tm[machines];
    //int t0[machines];

    //for all machine-pairs (reduce?) O(m^2) m*(m-1)/2
    for (l = 0; l < _somme_; l++) {
        
        i=l;//machinePairOrder[l];//start by most successful machine-pair....

        johnson_initCmax(machines, tmp, ma, i, machine, tempsMachines, minTempsArr, limite1);

        //johnson_heuristiqueCmax(tmp, ma, i);//compute johnson sequence //O(n)

        //ok
        johnson_heuristiqueCmax(jobs, tmp, ma, i, tabJohnson,tempsLag, job, tempsJob);//compute johnson sequence //O(n)

        if (nbAffectFin != 0) {
            //johnson_cmaxFin(tmp, ma);
            johnson_cmaxFin(tmp, ma, tempsMachinesFin);
        } else   {
            tmp[1] += minTempsDep[ma[1]];
        }

        //take max
        if (tmp[1] > moinsBon) {
//            
            bestind=i;
            moinsBon = tmp[1];
        }

        //early exit from johnson (if lb > best)
        if (moinsBon > minCmax && minCmax != -1) {
             valBorneInf[0] = moinsBon;
            //countMachinePairs[bestind]++;
            return bestind;
        }
//        }
    }
//    printf("%d\t%d\tJohnson elim\t%d \n",moinsBon,minCmax,somme);

//    countMachinePairs[bestind]++;
    //  nbborne++ ;
    valBorneInf[0] = moinsBon;
    return bestind;
} // 

void johnson_remplirLag(const int machines, const int jobs, int *machine, int *tempsLag, const int *tempsJob){
   
    int m1, m2;

    //for all jobs and all machine-pairs
    for (int i = 0; i < _somme_; i++) {
        m1 = machine[i];
        m2 = machine[_somme_ + i];
        for (int j = 0; j < jobs; j++) {
            tempsLag[i*jobs+j] = 0;
            //term q_iuv in Lageweg'78
            for (int k = m1 + 1; k < m2; k++)
                tempsLag[i*jobs+j] += tempsJob[k*jobs+j];
        }
        //BandH bound...
        //if(m1==0 && m2==machines-1){
//            printf("row %d\n",i);
        //    for(int k=0;k<jobs;k++){
//                printf("%4d ",tempsLag[i*_jobs_+k]);

          //      lagBandH[k]=tempsLag[i*_jobs_+k]+tempsJob[(i+1) * _jobs_ + k];
//                printf("%4d ",lagBandH[k]);
         //   }
        //}
    }
}



void johnson_remplirMachine(const int machines, int *machine )
{
    int cmpt = 0;
    //[0 0 0 ...  0  1 1 1 ... ... M-3 M-3 M-2 ]
    //[1 2 3 ... M-1 2 3 4 ... ... M-2 M-1 M-1 ]
    for (int i = 0; i < (machines - 1); i++) {
        for (int j = i + 1; j < machines; j++) {
            machine[cmpt]         = i;
            machine[(_somme_) + cmpt] = j; //(?????)

            //printf("%d,%d ",machine[cmpt],machine[somme_pad + cmpt]);
            cmpt++;
        }
    }
}


void johnson_remplirTabJohnson(const int machines, const int jobs,  int *tabJohnson, 
    int *tempsLag, const int *tempsJob ){

    int cmpt = 0;

    //for all machine-pairs compute Johnson's sequence
    for (int i = 0; i < (machines - 1); i++){
        for (int j = i + 1; j < machines; j++) {
            //(const int machines, const int jobs, int * ordo, int m1, int m2, 
            //int s,int *tempsLag, const int *tempsJob)
            //johnson_bound(tabJohnson[cmpt], i, j, cmpt);

            ////*ordo?
            johnson_bound(machines, jobs, tabJohnson+cmpt*jobs, i, j, cmpt, tempsLag, tempsJob);
        
            cmpt++;
        }
    }
}


int johnson_calculBorne(const int machines, const int jobs, int *job, int minCmax,
    int *minTempsArr, int *minTempsDep, int *machine, int *tempsMachines,int *tempsMachinesFin, 
    int *tempsLag, const int limite1, const int limite2, const int *tempsJob){

    int valBorneInf[2];

    // (const int machines, const int jobs, int *job, int *valBorneInf, int minCmax, 
    // int *minTempsArr, int *minTempsDep, int *machine, int *tempsMachines,int *tempsMachinesFin, 
    // int *tempsLag, const int *tempsJob)

    johnson_borneInfMakespan(machines, jobs, job, valBorneInf, minCmax, 
        minTempsArr, minTempsDep, machine,tempsMachines, tempsMachinesFin, tempsLag, limite1, limite2, tempsJob);

    return valBorneInf[0];
}





void johnson_set_tempsMachinesFin_tempsJobFin(const int machines, const int jobs, const int permutation[], int *tempsMachinesFin,
    const int limite2, const int *tempsJob){


    int jobCour;
    
    for (int j = machines - 1; j >= 0; j--) tempsMachinesFin[j]=0;

    for (int k = jobs - 1; k>=limite2; k--) {
        jobCour= permutation[k];

        tempsMachinesFin[machines-1]+=tempsJob[(machines-1)*jobs+jobCour];
        for (int j = machines - 2; j >= 0; j--){
            tempsMachinesFin[j]= max(tempsMachinesFin[j],tempsMachinesFin[j+1])+tempsJob[j*jobs+jobCour];
        } 
    }
}


void johnson_set_tempsMachines_retardDebut(const int machines, const int jobs, const int permutation[], int *tempsMachines, 
    const int limite1, const int limite2, const int *tempsJob){


    int job;
    for (int mm = 0; mm < machines; mm++){
        tempsMachines[mm] = 0;
    }
    for (int j = 0; j <= limite1; j++) {
        job = permutation[j];
        tempsMachines[0] = tempsMachines[0] + tempsJob[0*jobs+job]; ///???
        for (int m = 1; m < machines; m++){
            tempsMachines[m] = max(tempsMachines[m],
              tempsMachines[m - 1]) + tempsJob[m*jobs+job];
        }
    }

    int gap,minptm_rem;

    for(int j=1;j<machines;j++){
        gap = tempsMachines[j] - tempsMachines[j-1];
        minptm_rem=INT_MAX;
        for(int k=limite1+1;k<limite2;k++){
            job=permutation[k];
            if(tempsJob[(j-1)*jobs+job]<minptm_rem)minptm_rem=tempsJob[(j-1)*jobs+job];
        }
        if(minptm_rem>gap)
            tempsMachines[j]+=(minptm_rem-gap);
    }

//    printf("%d ",courant->tempsMachines[machines - 1]);
}



// void johnson_partial_cost(int machines, int jobs, int permutation[], int limit1, int limit2, 
//     int * couts, int from, int to, int *tempsJob){

//     int tmp[machines];

//     for (int mm = 0; mm < machines; mm++) tmp[mm] = 0;

//     int job;
//     for(int j=0;j<jobs;j++){
//         if(j<to || j>from)
//             job=permutation[j];
//         else if(j==to)
//             job=permutation[from];
//         else
//             job=permutation[j-1];

//         tmp[0]=tmp[0]+tempsJob[0*jobs+job];
//         for(int m=1;m<machines;m++) tmp[m]= max(tmp[m],tmp[m-1])+tempsJob[m*jobs+job];
//     }

//     couts[0]=tmp[machines-1];
// }


// void johnson_criteres_calculer(int machines, int jobs, int permutation[], 
//     int * cmax, int * tardiness, int *tempsJob){

//     int temps[machines];

//     for (int mm = 0; mm < machines; mm++) temps[mm] = 0;
//     *tardiness = 0;
//     for (int j = 0; j < jobs; j++) {
//         int job = permutation[j];
//         temps[0] = temps[0] + tempsJob[0*jobs+job]; ///????
//         for (int m = 1; m < machines; m++) temps[m] = max(temps[m], temps[m - 1]) + tempsJob[m*jobs+job];
//     }
//     *cmax = temps[machines - 1];
// }



int johnson_evalSolution(const int machines, const int jobs, const int *permutation, const int *tempsJob){

    int temps[machines];

    for (int mm = 0; mm < machines; mm++) temps[mm] = 0;
    for (int j = 0; j < jobs; j++) {
        int job = permutation[j];
        temps[0] = temps[0] + tempsJob[0*jobs+job];
        for (int m = 1; m < machines; m++) temps[m] = max(temps[m], temps[m - 1]) + tempsJob[m*jobs+job];
    }
    return temps[machines - 1]; // return makespan
}



int johnson_bornes_calculer(const int machines, const int jobs, int *job, const int permutation[], 
    const int limite1, const int limite2,  int minCmax,int *tempsMachines, int *tempsMachinesFin,
    int *minTempsArr, int *minTempsDep, int *machine, int *tempsLag, const int* tempsJob){
    
    //minCmax sometimes is related to as 'best'
    int r;

    if(limite1 == (limite2-1) ){
        return johnson_evalSolution(machines, jobs, permutation, tempsJob); //OK
    }

    //compute front --OK
    johnson_set_tempsMachines_retardDebut(machines, jobs, permutation, tempsMachines, 
        limite1, limite2, tempsJob);//OK
    //settting some values
    
    johnson_set_job_jobFin(jobs, permutation, limite1, limite2, job);//OK


    // johnson_set_nombres(jobs, limite1, limite2); //OK
    //compute tail

    //NEW FOR MCORE
    johnson_set_tempsMachinesFin_tempsJobFin(machines, jobs, permutation, tempsMachinesFin,
        limite2, tempsJob); //OK

    //johnson...
    //r=johnson_calculBorne(best);

// (const int machines, const int jobs, int *job, int minCmax,
//     int *minTempsArr, int *minTempsDep, int *machine, int *tempsMachines,int *tempsMachinesFin, 
//     int *tempsLag, const int *tempsJob)

    r = johnson_calculBorne(machines, jobs, job,minCmax,  
        minTempsArr, minTempsDep, machine, tempsMachines, tempsMachinesFin,tempsLag,limite1, limite2, tempsJob);//OK
    // printf("\nborneInfCMax: %d\n.",r);

    return r;

}





void johnson_bound_search(const int machines, const int jobs, const int *tempsJob){


    register unsigned int flag = 0;
    register int bit_test = 0;
    register int i, depth; 
    register unsigned long long int local_tree = 0ULL;
    int num_sol = 0;
   //register int custo = 0;
    int lowerbound = 0;
    int incumbent = 1713;
    register int p1;


    int tempsMachinesFin[machines]; //front
    int tempsMachines[machines]; //back
    int job[_MAX_J_JOBS_];
    //search
    int permutation[jobs];
    int scheduled[jobs];
    int position[jobs];


    johnson_remplirMachine(machines, machine);//what is the vector to pass?
    remplirTempsArriverDepart(minTempsArr,minTempsDep, machines, jobs, tempsJob); //verificar se eh o mesmo do outro bound
    johnson_remplirLag(machines, jobs, machine, tempsLag,tempsJob);
    johnson_remplirTabJohnson(machines, jobs, tabJohnson, tempsLag, tempsJob);


    start_vector(permutation,jobs);
    start_vector(position,jobs);

    /*Inicializacao*/
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
                                        
                    // lowerbound = johnson_bornes_calculer(machines, jobs, permutation, 
                    //     depth, jobs,incumbent);

                    lowerbound = johnson_bornes_calculer(machines, jobs, job, permutation,
                        depth, jobs, incumbent, tempsMachines, tempsMachinesFin, minTempsArr,
                        minTempsDep, machine, tempsLag,tempsJob);

                    if(lowerbound<incumbent){//is it feasible

                        flag |= (1ULL<<scheduled[depth]);
                        depth++;
                        local_tree++;

                        if(depth == jobs && lowerbound < incumbent){ //handle solution
                            
                            num_sol++;
                            incumbent = lowerbound;
                            printf("\n");
                            printf("\nnew incumbent solution: %d.\n",lowerbound);
                            print_permutation(permutation,jobs);
                        
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


void johnson_call_search(short p){

    int jobs;
    int machines;
    int *times = get_instance(&machines,&jobs,p);
    print_instance(machines, jobs, times);
    johnson_bound_search(machines, jobs, times);

}///////////////
