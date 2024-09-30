#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>
#include <climits>
#include <math.h>
#include <algorithm>



#define _EMPTY_      -1

#define MAX_BOARD 256

unsigned beauty[MAX_BOARD*MAX_BOARD];

#define beauty(i,j) beauty[(i)*(size)+(j)]

int num_sols = 0;

//!!!!!!!! ALWAYS USE -1
int beauty16[] = {9, 11, 4, 14, 10, 2, 5, 1, 16, 12, 15, 7, 3, 13, 6, 8};
//!!!!!!!! ALWAYS USE -1
int beauty32[] = {17, 14, 12, 23, 18, 26, 6, 11, 13, 4, 25, 30, 24, 31, 5, 1, 32, 28, 2, 9, 3, 8, 29, 20, 22, 27, 7, 15, 10, 21, 19, 16};
//!!!!!!!! ALWAYS USE -1
int beauty48[] = {25, 27, 29, 26, 17, 34, 12, 38, 9, 18, 30, 21, 7, 43, 36, 5, 14, 45, 37, 3, 6, 2, 39, 1, 48, 41, 47, 42, 46, 8, 4, 11, 44, 15, 19, 32, 28, 33, 40, 10, 23, 13, 35, 16, 31, 20, 22, 24};
//!!!!!!!! ALWAYS USE -1
int beauty64[] = {33, 35, 37, 34, 40, 42, 36, 20, 17, 48, 13, 52, 12, 27, 38, 55, 43, 39, 8, 24, 58, 44, 6, 15, 5, 21, 51, 3, 7, 2, 4, 1, 64, 61, 63, 50, 62, 56, 16, 60, 47, 59, 22, 18, 46, 57, 9, 41, 10, 26, 54, 11, 53, 14, 49, 19, 45, 29, 23, 25, 31, 28, 30, 32};
////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////


inline void queens_return_beauty_from_sol(int *__restrict__ board, unsigned * beauty_vector, const int size){
    
    for(int i = 0; i<size;++i){
        beauty_vector[i] = beauty(i,board[i]);
        //printf("%u - ", beauty_vector[i]);
    }

    std::sort(beauty_vector, beauty_vector+size,std::greater<>());

}   

inline void queens_beauty_partial_sol(int *__restrict__ board, unsigned * beauty_vector, const int size, const int position, bool sort_sol){
    
    beauty_vector[position] = beauty(position,board[position]);
    //printf("%u - ", beauty_vector[i]);
    
    if(sort_sol){
        std::sort(beauty_vector, beauty_vector+(position+1),std::greater<>());
    }

}   

bool queens_partial_is_more_beautiful(unsigned *__restrict__ current_beauty_vector, unsigned *__restrict__ best_beauty_vector, const int size){
    
    for(int i = 0; i<size;++i){
        if(current_beauty_vector[i] > best_beauty_vector[i])
            return false; //it is not more beautiful
        else{
            if(current_beauty_vector[i] < best_beauty_vector[i])
                return true; //it is
            else
                continue; //continue to decide
        }
    }
    return false; //if it is equal goes up to this point
}   

void queens_start_board_beauty(const int size){

    for(int i = 0; i<size;++i){
        for(int j = 0; j<size;++j){
            beauty(i,j) = (pow((2*(i+1)-size-1),2) + pow((2*(j+1)-size-1),2));
        }
    }

    for(int i = 0; i<size;++i){
        for(int j = 0; j<size;++j){
            printf(" %d - ",beauty(i,j));
        }
        printf("\n");
    }
    printf("\n");
}



/////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
// Check if placing a queen at board[row] in column `col` causes any conflicts
inline int is_safe(int *__restrict__ board, const int row, const int col) {
    for (int prev_col = 0; prev_col < col; prev_col++) {
        int prev_row = board[prev_col];
        if (prev_row == row || abs(prev_row - row) == abs(prev_col - col)) {
            return 0;  // Conflict found
        }
    }
    return 1;  // Safe to place the queen
}

// Shuffle an array of row indices
inline void shuffle(int *__restrict__ array, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

// Function to construct a valid N-Queens solution with random row selection
int construct_random_solution(int *__restrict__ board, unsigned *__restrict__ current_beauty,
    unsigned *__restrict__ best_beauty , int n) {
   
    for (int col = 0; col < n; col++) {
        
        int rows[MAX_BOARD];  // Array to hold possible rows for this column
        int row_count = 0;

        // Generate a list of safe rows for this column
        for (int row = 0; row < n; row++) {
            if (is_safe(board, row, col)) {
                rows[row_count++] = row;
            }
        }

        // If no valid row exists for this column, restart the solution
        if (row_count == 0) {
            return 0;  // Failed to place a queen in this column, restart
        }

        // Shuffle the list of valid rows and choose a random one
        shuffle(rows, row_count);
        board[col] = rows[0];  // Place the queen in a random safe row
/* 
        if((col%5)==0){
            queens_beauty_partial_sol(board, current_beauty,n,col, true);
            if(!queens_partial_is_more_beautiful(current_beauty, best_beauty, col+1)){
                return 0;
            }
        }
        else{
            queens_beauty_partial_sol(board, current_beauty,n,col, false);
        }
            
 */
    
       
    }

    return 1;  // Successfully placed all queens
}


// Function to solve the N-Queens problem with random valid solutions
bool solve_n_queens(int *best, unsigned * best_beauty, int *current, unsigned * current_beauty, int n) {

   
    bool success = false;
    
    // Keep trying until a valid solution is found
    while (!success) {
        success = construct_random_solution(current, current_beauty,best_beauty, n);
    }
   
   
    queens_return_beauty_from_sol(current, current_beauty, n);
    

     if(queens_partial_is_more_beautiful(current_beauty, best_beauty, n)){
        printf("\nNew solution found: \n");
        printf("From best: \n");
        for(int i = 0;i<n;++i){
            printf(" %d - ", best[i]);
        }
        printf("\nBest-Beauty vector: \n");
        for(int i = 0; i<n;++i){
            printf(" %u - ", best_beauty[i]);
        }
        printf("\n");

        fflush( stdout );
        printf("\nto current:");
        for(int i = 0;i<n;++i){
            printf(" %d - ", current[i]);
        }
        printf("\nBeauty vector: \n");
        for(int i = 0; i<n;++i){
            printf(" %u - ", current_beauty[i]);
        }
        printf("\n");

        return true;
    }
    return false;
   
    //printf("\n");
}

int main(int argc, char *argv[]) {
   setbuf(stdout, NULL);
    int n;
    int number_execs;
    // Seed the random number generator
    srand(time(NULL));
    n = atoi(argv[1]);
    number_execs = atoi(argv[2]);

    // Input the size of the board (number of queens)
    printf("Enter the number of queens: %d\n", n);
    // Solve the N-Queens problem

    int *best = (int*)malloc(sizeof(int)*n);
    unsigned *best_beauty = (unsigned*)malloc(sizeof(unsigned)*n);
    int *current = (int*)malloc(sizeof(int)*n);
    unsigned *current_beauty = (unsigned*)malloc(sizeof(unsigned)*n);

    for(int i = 0; i<n;++i){
         best[i] = INT_MAX;
         best_beauty[i] = UINT_MAX;
         current_beauty[i] = UINT_MAX;
    }
       
    int *tmp;
    unsigned *tmp_beauty;

    //Initialize the beauty board for a given size
    queens_start_board_beauty(n);
 /* 
    printf("\nBeauty 64:");
    for(int i = 0;i<n;++i){
        beauty64[i] = beauty64[i]-1;
        printf(" %d - ", beauty64[i]);
    }
    queens_return_beauty_from_sol(beauty64, current_beauty, n);
    printf("\nBeauty vector: \n");
    for(int i = 0; i<n;++i){
        printf(" %u - ", current_beauty[i]);
    }
    printf("\n");

    
    exit(1); */
  
  // best[0] = 0;
   //best[1] = 2;
    for(int i = 0; i<number_execs;++i){
        bool could_improve = solve_n_queens(best, best_beauty, current, current_beauty,n);
        if(could_improve){
            tmp = best;
            best = current;
            current = tmp;

            tmp_beauty = best_beauty;
            best_beauty = current_beauty;
            current_beauty = tmp_beauty;
        }
    }

    //solve_n_queens(best, current, n);
       
    return 0;
}