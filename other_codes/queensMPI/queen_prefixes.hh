#ifndef QUEEN_PREF_HH
#define QUEEN_PREF_HH



#include "queen_root_prefix.hh"
#include "macros.hh"


void inline prefixesHandleSol(char *path,unsigned int *vector_of_flags, unsigned int flag,short *board,int initialDepth,int num_sol){


   printf("\n Solucao %d. \n\tVetor: ",num_sol);
    for(int i = 0; i<initialDepth;++i){

      printf("%d ", board[i] );

    }
    printf("\n");
    vector_of_flags[num_sol] = flag;
    for(int i = 0; i<initialDepth;++i)
        path[num_sol*initialDepth+i] = (char)board[i];
    //printf("\n");
    //exit(1);
}


inline void handleSolution(short *board,int size){

    int coluna;

   // ++NSOL;

    printf("\n Vetor: ");
    for(int i = 0; i<size;++i){

      printf("%d ", board[i] );
    }
    printf("\n");
    //printf("\n");
    //exit(1);

  for(int i =0; i<size;++i){
      coluna = board[i];

      for(int j = 0; j<size;++j){
          if(j==coluna)
              printf(" %s ", "Q");
          else
              printf(" %c ", 35);
      }
      printf("\n");
  }

}




inline bool stillLegal(const short *board, const int r)
{
  
  register int i;
  register int ld;
  register int rd;
  // // Check vertical
  for ( i = 0; i < r; ++i)
    if (board[i] == board[r]) return false;
  //  Check diagonals
    ld = board[r];  //left diagonal columns
    rd = board[r];  // right diagonal columns
    for ( i = r-1; i >= 0; --i) {
      --ld; ++rd;
      if (board[i] == ld || board[i] == rd) return false;
    }

    return true;
}

inline bool MCstillLegal(const char *board, const int r)
{
  register int i;
  register int ld;
  register int rd;
  // Check vertical
  for ( i = 0; i < r; ++i)
    if (board[i] == board[r]) return false;
    // Check diagonals
    ld = board[r];  //left diagonal columns
    rd = board[r];  // right diagonal columns
    for ( i = r-1; i >= 0; --i) {
      --ld; ++rd;
      if (board[i] == ld || board[i] == rd) return false;
    }

    return true;
}




inline void prefixesHandleSol(QueenRoot *root_prefixes,unsigned int flag,char *board,int initialDepth,int num_sol){

    root_prefixes[num_sol].flag = flag;

    for(int i = 0; i<initialDepth;++i)
      root_prefixes[num_sol].board[i] = (char)board[i];
}


unsigned int BP_queens_prefixes(int size, int initialDepth ,unsigned long long *tree_size, QueenRoot *root_prefixes){

    register unsigned int flag = 0;
    register int bit_test = 0;
    register char vertice[20]; //representa o ciclo
    register int i, nivel; //para dizer que 0-1 ja foi visitado e a busca comeca de 1, bote 2
    register unsigned long long int local_tree = 0ULL;
    unsigned int num_sol = 0;
   //register int custo = 0;

    /*Inicializacao*/
    for (i = 0; i < size; ++i) { //
        vertice[i] = -1;
    }

    nivel = 0;

    do{

        vertice[nivel]++;
        bit_test = 0;
        bit_test |= (1<<vertice[nivel]);


        if(vertice[nivel] == size){
            vertice[nivel] = _VAZIO_;
                //if(block_ub > upper)   block_ub = upper;
        }else if ( MCstillLegal(vertice, nivel) && !(flag &  bit_test ) ){ //is legal

                flag |= (1ULL<<vertice[nivel]);
                nivel++;
                ++local_tree;
                if (nivel == initialDepth){ //handle solution
                   prefixesHandleSol(root_prefixes,flag,vertice,initialDepth,num_sol);
                   num_sol++;
            }else continue;
        }else continue;

        nivel--;
        flag &= ~(1ULL<<vertice[nivel]);

    }while(nivel >= 0);

    *tree_size = local_tree;

    return num_sol;
}




inline unsigned int BP_queens_prefixes(int size, int initialDepth ,unsigned long long *tree_size,char *path,unsigned int *vector_of_flags){

    register unsigned int flag = 0;
    register int bit_test = 0;
    register short vertice[MAX]; //representa o ciclo
    register int i, nivel; //para dizer que 0-1 ja foi visitado e a busca comeca de 1, bote 2
    register unsigned long long int local_tree = 0ULL;
    unsigned int num_sol = 0;
   //register int custo = 0;

    /*Inicializacao*/
    for (i = 0; i < size; ++i) { //
        vertice[i] = -1;
    }

    nivel = 0;

    do{

        vertice[nivel]++;
        bit_test = 0;
        bit_test |= (1<<vertice[nivel]);


        if(vertice[nivel] == size){
            vertice[nivel] = _VAZIO_;
                //if(block_ub > upper)   block_ub = upper;
        }else if ( stillLegal(vertice, nivel) && !(flag &  bit_test ) ){ //is legal

                flag |= (1ULL<<vertice[nivel]);
                nivel++;
                ++local_tree;
                if (nivel == initialDepth){ //handle solution
                   prefixesHandleSol(path,vector_of_flags,flag,vertice,initialDepth,num_sol);
                   num_sol++;
            }else continue;
        }else continue;

        nivel--;
        flag &= ~(1ULL<<vertice[nivel]);

    }while(nivel >= 0);

    *tree_size = local_tree;

    return num_sol;
}





//this one is for profiling tool
void BP_queens_prefixes_analisys(int initialDepth, int size, unsigned int flag, short *vertice, unsigned long long *tree_size,int *tree_analisys){

    // register unsigned int flag = 0;
    register int bit_test = 0;
    register int nivel; //para dizer que 0-1 ja foi visitado e a busca comeca de 1, bote 2
    register unsigned long long int local_tree = 0ULL;

   //register int custo = 0;

    // /*Inicializacao*/
    // for (i = 0; i < size; ++i) { //
    //     vertice[i] = -1;
    // }

    nivel = initialDepth;

    do{

        vertice[nivel]++;
        bit_test = 0;
        bit_test |= (1<<vertice[nivel]);


        if(vertice[nivel] == size){
            vertice[nivel] = _VAZIO_;
                //if(block_ub > upper)   block_ub = upper;
        }else 
          if ( stillLegal(vertice, nivel) && !(flag &  bit_test ) ){ //is legal

                flag |= (1ULL<<vertice[nivel]);
                nivel++;
                ++local_tree;
                tree_analisys[nivel]++;

                if (nivel == size){ //handle solution
                //    prefixesHandleSol(path,vector_of_flags,flag,vertice,initialDepth,num_sol);
                //    num_sol++;
               }else continue;
        }else continue;

        nivel--;
        flag &= ~(1ULL<<vertice[nivel]);

    }while(nivel >= initialDepth);

    *tree_size += local_tree;
}


unsigned int BP_queens_prefixes_analisys(int size, int initialDepth ,unsigned long long *tree_size,char *path,unsigned int *vector_of_flags, unsigned long long *tree_analisys){

    register unsigned int flag = 0;
    register int bit_test = 0;
    register short vertice[MAX]; //representa o ciclo
    register int i, nivel; //para dizer que 0-1 ja foi visitado e a busca comeca de 1, bote 2
    register unsigned long long int local_tree = 0ULL;
    unsigned  int num_sol = 0;
   //register int custo = 0;

    /*Inicializacao*/
    for (i = 0; i < size; ++i) { //
        vertice[i] = -1;
    }

    nivel = 0;

    do{

        vertice[nivel]++;
        bit_test = 0;
        bit_test |= (1<<vertice[nivel]);


        if(vertice[nivel] == size){
            vertice[nivel] = _VAZIO_;
                //if(block_ub > upper)   block_ub = upper;
        }else if ( stillLegal(vertice, nivel) && !(flag &  bit_test ) ){ //is legal

                flag |= (1ULL<<vertice[nivel]);
                nivel++;
                ++local_tree;
                tree_analisys[nivel]++;

                if (nivel == initialDepth){ //handle solution
                   prefixesHandleSol(path,vector_of_flags,flag,vertice,initialDepth,num_sol);
                   num_sol++;
                }else continue;
        }else continue;

        nivel--;
        flag &= ~(1ULL<<vertice[nivel]);

    }while(nivel >= 0);

    *tree_size = local_tree;

    return num_sol;
}


#endif