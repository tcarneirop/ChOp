#ifndef MC_QUEENS_DFS_HH
#define MC_QUEENS_DFS_HH


#include "macros.hh"
#include "queen_root_prefix.hh"



void BP_queens_MC_root_dfs(int N, int id, unsigned int nPreFixos,int nivelPreFixos,QueenRoot *root_prefixes,int*vector_of_tree_size, int *sols){


        register int idx = id;
        register unsigned int flag = 0;
        register unsigned int bit_test = 0;
        register char vertice[20]; //representa o ciclo
        register int N_l = N;
        register int i, nivel; //para dizer que 0-1 ja foi visitado e a busca comeca de 1, bote 2
        register int qtd_solucoes_thread = 0;
//  register int UB_local;
        register int nivelGlobal = nivelPreFixos;
        register int tree_size = 0;

       for (i = 0; i < N_l; ++i) {
            vertice[i] = _VAZIO_;
        }

       flag = root_prefixes[idx].flag;

        #pragma unroll
        for (i = 0; i < nivelGlobal; ++i) {

            vertice[i] = root_prefixes[idx].board[i];

        }
        nivel=nivelGlobal;

        do{

            vertice[nivel]++;
            bit_test = 0;
            bit_test |= (1<<vertice[nivel]);

            if(vertice[nivel] == N_l){
                vertice[nivel] = _VAZIO_;
                //if(block_ub > upper)   block_ub = upper;
            }else if (MCstillLegal(vertice, nivel) && !(flag &  bit_test )){

                    
                    ++tree_size;
                    flag |= (1ULL<<vertice[nivel]);

                    nivel++;

                    if (nivel == N_l) { //sol
                        ++qtd_solucoes_thread; 
                    }else continue;
                }else continue;

            nivel--;
            flag &= ~(1ULL<<vertice[nivel]);

            }while(nivel >= nivelGlobal); //FIM DO DFS_BNB

        //root_prefixes[position].sols = qtd_solucoes_thread;
        //root_prefixes[position].melhorSol = upper;
        //root_prefixes[position].tree_size= tree_size;
        sols[idx] = qtd_solucoes_thread;
        vector_of_tree_size[idx] = tree_size;

}//kernel
////////


void queens_mcore_BP_dfs(int size,int id,unsigned int *vector_of_flags, char *preFixos_d, int nPreFixos, int nivelPrefixo, int *sols_d, int*vector_of_tree_size){


    register int idx = id;

    // if (idx < nPreFixos) { //INICIO DO DFS_BNB

        register unsigned int flag = 0;
        register unsigned int bit_test = 0;
        register char vertice[20]; //representa o ciclo
        register int N_l = size;
        register int i, nivel,j; //para dizer que 0-1 ja foi visitado e a busca comeca de 1, bote 2
        register int qtd_solucoes_thread = 0;
//  register int UB_local;
        register int nivelGlobal = nivelPrefixo;
        register int tree_size = 0;

       for (i = 0; i < N_l; ++i) {
            vertice[i] = _VAZIO_;
        }

       flag = vector_of_flags[idx];

        #pragma unroll
        for (i = 0, j = idx * nivelGlobal; i < nivelGlobal; ++i) {

        	vertice[i] = preFixos_d[j + i];

        }

        nivel=nivelGlobal;

        do{

            vertice[nivel]++;
            bit_test = 0;
            bit_test |= (1<<vertice[nivel]);


            if(vertice[nivel] == N_l){
                vertice[nivel] = _VAZIO_;
                //if(block_ub > upper)   block_ub = upper;
            }else if ( MCstillLegal(vertice, nivel) && !(flag &  bit_test ) ){

                    ++tree_size;
                    flag |= (1ULL<<vertice[nivel]);
                    nivel++;

                    if (nivel == N_l) { //sol
                        ++qtd_solucoes_thread;
                    }//fi
                    else continue;
                }else continue;

            nivel--;
            flag &= ~(1ULL<<vertice[nivel]);

        }while(nivel >= nivelGlobal); //FIM DO DFS_BNB

        sols_d[idx] = qtd_solucoes_thread;
        vector_of_tree_size[idx] = tree_size;

    // }

}//kernel
////////
#endif