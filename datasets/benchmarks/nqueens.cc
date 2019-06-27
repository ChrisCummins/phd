// The n queens puzzle.
// Copyright 2017 Sol from https://solarianprogrammer.com.
// Released under GPLv3.
// See <https://github.com/sol-prog/N-Queens-Puzzle>

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Store the puzzle (problem) size and the number of valid solutions.
typedef struct {
  int size;
  int solutions;
} NQueens;

// Show the queens positions on the board in compressed form,
// each number represent the occupied column position in the corresponding row.
void show_short_board(NQueens *nqueens, int *positions) {
  for (int i = 0; i < nqueens->size; ++i) {
    printf("%d ", positions[i]);
  }
  printf("\n");
}

// Show the full NxN board
void show_full_board(NQueens *nqueens, int *positions) {
  for (int row = 0; row < nqueens->size; ++row) {
    for (int column = 0; column < nqueens->size; ++column) {
      if (positions[row] == column) {
        printf("Q ");
      } else {
        printf(". ");
      }
    }
    printf("\n");
  }
  printf("\n");
}

// Check if a given position is under attack from any of
// the previously placed queens (check column and diagonal positions)
bool check_place(int *positions, int ocuppied_rows, int column) {
  for (int i = 0; i < ocuppied_rows; ++i) {
    if (positions[i] == column || positions[i] - i == column - ocuppied_rows ||
        positions[i] + i == column + ocuppied_rows) {
      return false;
    }
  }
  return true;
}

// Try to place a queen on target_row by checking all N possible cases.
// If a valid place is found the function calls itself trying to place a queen
// on the next row until all N queens are placed on the NxN board.
void put_queen(NQueens *nqueens, int *positions, int target_row) {
  // Base (stop) case - all N rows are occupied
  if (target_row == nqueens->size) {
    // show_short_board(nqueens, positions);
    show_full_board(nqueens, positions);
    nqueens->solutions++;
  } else {
    // For all N columns positions try to place a queen
    for (int column = 0; column < nqueens->size; ++column) {
      // Reject all invalid positions
      if (check_place(positions, target_row, column)) {
        positions[target_row] = column;
        put_queen(nqueens, positions, target_row + 1);
      }
    }
  }
}

// Solve the n queens puzzle and print the number of solutions
void solve(NQueens *nqueens) {
  int *positions = (int *)malloc(nqueens->size * sizeof(int));
  memset(positions, ~0, nqueens->size * sizeof(int));
  put_queen(nqueens, positions, 0);
  printf("Found %d solutions\n", nqueens->solutions);
  free(positions);
}

int main(void) {
  NQueens nqueens = {.size = 8};
  solve(&nqueens);
  return 0;
}
