#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <limits.h>

#define MAXN 256  // Maximum board size (can be adjusted as needed)

//int first[] = {1, 3, 5, 2, 4, 9, 11, 13, 15, 6};
int first[] = {0, 2, 4, 1, 3, 8, 10, 12, 14, 5};
int first_big[] = {1, 3, 5, 2, 4, 9, 11, 13, 15, 6, 8, 19, 7, 22, 10, 25, 27, 29, 31, 12};
int num_sols = 0;
int FIRST_PREFIX = 7;

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAXN 256  // Maximum board size (can be adjusted as needed)

bool is_an_earlier_sol(int *__restrict__ best, int *__restrict__ current, const int n){

    for(int i = 0; i < n; ++i){
        if(current[i] < best[i]){
            return true;
        }
        else{
            if(current[i] > best[i])
                return false;
            else 
                continue;
        }
    }
    return false;
}

// Check if placing a queen at board[row] in column `col` causes any conflicts
int is_safe(int *__restrict__ board, const int row, const int col) {
    for (int prev_col = 0; prev_col < col; prev_col++) {
        int prev_row = board[prev_col];
        if (prev_row == row || abs(prev_row - row) == abs(prev_col - col)) {
            return 0;  // Conflict found
        }
    }
    return 1;  // Safe to place the queen
}

// Shuffle an array of row indices
void shuffle(int *__restrict__ array, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

// Function to construct a valid N-Queens solution with random row selection
int construct_random_solution(int *__restrict__ board, int *__restrict__ best, int n) {

   
    for(int i = 0; i<FIRST_PREFIX;++i){
        board[i] = first_big[i]-1;
    }

    for (int col = FIRST_PREFIX; col < n; col++) {
        
        int rows[MAXN];  // Array to hold possible rows for this column
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
       
       if(col%5 && !is_an_earlier_sol(best,board,col)){
            return 0;
        };
    }

    return 1;  // Successfully placed all queens
}

// Function to solve the N-Queens problem with random valid solutions
bool solve_n_queens(int *best, int *current, int n) {

   
    int success = 0;
    int *temp;

    // Keep trying until a valid solution is found
    while (!success) {
        success = construct_random_solution(current, best, n);
    }

    if(is_an_earlier_sol(best,current,n)){
        printf("\nNew solution found: \n");
       // printf("From best: ");
       // for(int i = 0;i<n;++i){
       //     printf(" %d - ", best[i]);
       // }
        fflush( stdout );
       // printf("\nto current:");
        for(int i = 0;i<n;++i){
            printf(" %d - ", current[i]);
        }
        return true;
    }
    return false;
    // Print the valid random solution
    //printf("Random valid solution for %d-Queens:\n", n);
    //printf("\n");
    //printf("\n");
   
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
    int *current = (int*)malloc(sizeof(int)*n);;

    for(int i = 0; i<n;++i)
        best[i] = INT_MAX;

    int *tmp;
   
   best[0] = 0;
   //best[1] = 2;

    for(int i = 0; i<number_execs;++i){
        bool could_improve = solve_n_queens(best, current, n);
        if(could_improve){
            tmp = best;
            best = current;
            current = tmp;
        }
    }
    

    return 0;
}