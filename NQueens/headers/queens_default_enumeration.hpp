
#ifndef QUEENS_GEN_ENUMERATION_HPP
#define QUEENS_GEN_ENUMERATION_HPP


#if defined(_OPENMP) && defined(ENABLE_OMP_OFFLOAD)
    #pragma omp declare target
#endif


void queens_default_subtree_enumeration(const unsigned idx, const int N, const unsigned nPrefixes, 
    const int initial_depth, QueenRoot *__restrict__ root_prefixes,
    unsigned long long int *__restrict__ vector_of_tree_size, 
    unsigned long long int *__restrict__ sols){


    unsigned int flag = 0;
    int8_t board[MAX_SIZE];
    int N_l = N;
    int i, depth;
    unsigned long long  qtd_sols_thread = 0ULL;
    int depthGlobal = initial_depth;
    unsigned long long int tree_size = 0ULL;

    for (i = 0; i < N_l; ++i) {
        board[i] = EMPTY;
    }

    flag = root_prefixes[idx].control;

    for (i = 0; i < depthGlobal; ++i)
        board[i] = root_prefixes[idx].board[i];

    depth=initial_depth;

    do{

        //board[depth]++;
        const int mask = 1 << ++board[depth];

        if(board[depth] == N_l){

            board[depth] = EMPTY;
            //--depth;
            flag &= ~(1<<board[--depth]);
        }
        else{

            if (!(flag &  mask ) && queens_is_legal_placement(board, depth)){

                ++tree_size;
                flag |= mask;

                ++depth;

                if (depth == N_l) { //sol
                    ++qtd_sols_thread ;
                    --depth;
                    flag &= ~mask;
                }//if 
            }//else if
        }


    }while(depth >= depthGlobal); 

    sols[idx] = qtd_sols_thread ;
    vector_of_tree_size[idx] = tree_size;

}//kernel
#if defined(_OPENMP) && defined(ENABLE_OMP_OFFLOAD)
    #pragma omp end declare target
#endif


#endif