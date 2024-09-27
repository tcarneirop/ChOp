#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAXN 256  // Maximum board size (can be adjusted as needed)

// Check if placing a queen at board[row] in column `col` causes any conflicts
int is_safe(int *__restrict__ board, int row, int col) {
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
int construct_random_solution(int *__restrict__ board, int n) {
    for (int col = 0; col < n; col++) {
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
    }

    return 1;  // Successfully placed all queens
}

// Function to solve the N-Queens problem with random valid solutions
void solve_n_queens(int n) {
    int board[MAXN];
    int success = 0;

    // Keep trying until a valid solution is found
    while (!success) {
        success = construct_random_solution(board, n);
    }

    // Print the valid random solution
    //printf("Random valid solution for %d-Queens:\n", n);
    //printf("\n");
    //printf("\n");
    for(int i = 0; i<n;++i){
        printf(" %d -  ", board[i]);
    } 
    //printf("\n");

}

int main(int argc, char *argv[]) {
    int n;

    // Seed the random number generator
    srand(time(NULL));
    n = atoi(argv[1]);
    // Input the size of the board (number of queens)
    printf("Enter the number of queens: %d\n", n);
    // Solve the N-Queens problem
    solve_n_queens(n);

    return 0;
}