// You are given a matrix, M. Write a function to determine if the matrix is
// diagonally the same.  E.g.
//
//   [1 2 3 4]
//   [5 1 2 3]
//   [6 5 1 2]
//   [7 6 5 1]
//
#include <vector>
#include "labm8/cpp/test.h"

using std::min;
using std::vector;

template <typename T>
bool IsDiagEquiv(const vector<vector<T>>& M) {
  // Tiling size.
  const size_t bj = (M.size() / 4) + 1;
  const size_t bi = (M.size() / 4) + 1;

  // Since we're tiling and parallelizing we can't have return statements inside
  // the inner worker, so store their values here.
  bool is_left_diag = true;
  bool is_right_diag = true;

  if (!M.size()) {
    return true;
  }
  const size_t n = M[0].size();

// Tile the outer execution loop.
#pragma omp parallel for collapse(2)
  for (size_t jj = 1; jj < M.size(); jj += bj) {
    for (size_t ii = 0; ii < n; ii += bi) {
      size_t uj = min(jj + bj, M.size());
      size_t ui = min(ii + bi, M[jj].size());

      // Inner loop executed by each worker.
      for (size_t j = jj; j < uj; ++j) {
        for (size_t i = ii; i < ui; ++i) {
          // Check diagonal from top left.
          if (i) {
            if (M[j][i] != M[j - 1][i - 1]) {
              is_left_diag = false;
            }
          }

          // Check diagonal from top right.
          if (i < n - 1) {
            if (M[j][i] != M[j - 1][i + 1]) {
              is_right_diag = false;
            }
          }
        }
      }
    }
  }

  return is_left_diag || is_right_diag;
}

// Tests:

// [] -> true
TEST(IsDiagEquiv, EmptyMatrix) {
  vector<vector<int>> M;
  EXPECT_EQ(IsDiagEquiv(M), true);
}

// [1] -> true
TEST(IsDiagEquiv, SingleElementMatrix) {
  vector<vector<int>> M{{1}};
  EXPECT_EQ(IsDiagEquiv(M), true);
}

// [1 2 3] -> true
TEST(IsDiagEquiv, Vector) {
  vector<vector<int>> M{{1, 2, 3}};
  EXPECT_EQ(IsDiagEquiv(M), true);
}

// [1 2 3]
// [0 1 2] -> true
TEST(IsDiagEquiv, TwoRowMatrix) {
  vector<vector<int>> M{{1, 2, 3}, {0, 1, 2}};
  EXPECT_EQ(IsDiagEquiv(M), true);
}

// [1 2 3]
// [1 2 3] -> false
TEST(IsDiagEquiv, TwoRowInvalidMatrix) {
  vector<vector<int>> M{{1, 2, 3}, {1, 2, 3}};
  EXPECT_EQ(IsDiagEquiv(M), false);
}

// [1 2 3 4]
// [5 1 2 3]
// [6 5 1 2]
// [7 6 5 1] -> true
TEST(IsDiagEquiv, ExampleInput) {
  vector<vector<int>> M{{1, 2, 3, 4}, {5, 1, 2, 3}, {6, 5, 1, 2}, {7, 6, 5, 1}};
  EXPECT_EQ(IsDiagEquiv(M), true);
}

// [1 2 3 4 5 6 7 8]
// [5 1 2 3 4 5 6 7]
// [6 5 1 2 3 4 5 6]
// [7 6 5 1 2 3 4 5]
// [8 7 6 5 1 2 3 4]
// [9 8 7 6 5 1 2 3] -> true
TEST(IsDiagEquiv, BiggerInput) {
  vector<vector<int>> M{{1, 2, 3, 4, 5, 6, 7, 8}, {5, 1, 2, 3, 4, 5, 6, 7},
                        {6, 5, 1, 2, 3, 4, 5, 6}, {7, 6, 5, 1, 2, 3, 4, 5},
                        {8, 7, 6, 5, 1, 2, 3, 4}, {9, 8, 7, 6, 5, 1, 2, 3}};
  EXPECT_EQ(IsDiagEquiv(M), true);
}

// [1 2 3]
// [2 3 1]
// [3 1 2] -> true
TEST(IsDiagEquiv, RightDiagonal) {
  vector<vector<int>> M{
      {1, 2, 3},
      {2, 3, 1},
      {3, 1, 2},
  };
  EXPECT_EQ(IsDiagEquiv(M), true);
}

// [1 1 3]
// [2 5 1]
// [4 1 2] -> true
TEST(IsDiagEquiv, NotDiagnoal) {
  vector<vector<int>> M{
      {1, 1, 3},
      {2, 5, 1},
      {4, 1, 2},
  };
  EXPECT_EQ(IsDiagEquiv(M), false);
}

TEST_MAIN();
