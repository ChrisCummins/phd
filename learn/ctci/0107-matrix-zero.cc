/*
 * Write an algorithm such that if an element in an MxN matrix 0, its
 * entire row and column are set to 0.
 */
#include "./ctci.h"

#include <unordered_set>

static unsigned int seed = 0xCEC;

//
// Perform two passes. The first, to record the row and column indices
// to zero. The second, to perform the actual zeroing.
//
// O(n^2) time, O(n) space.
//
template <typename T>
void matrix_zero(T *const m, const size_t rows, const size_t cols) {
  if (!(rows * cols) || !m) return;

  // Determine which rows and columns to zero.
  std::unordered_set<size_t> rows_s, cols_s;
  for (size_t i = 0; i < rows * cols; i++) {
    if (m[i] == 0) {
      auto x = i % cols;
      auto y = i / cols;

      rows_s.insert(y);
      cols_s.insert(x);
    }
  }

  for (auto &row : rows_s)
    for (size_t i = row * cols; i < (row + 1) * cols; i++) m[i] = 0;

  for (auto &col : cols_s)
    for (size_t i = col % cols; i < rows * cols; i += cols) m[i] = 0;
}

///////////
// Tests //
///////////

TEST(Permutation, matrix_zero) {
  int m1[] = {1, 2, 3, 4, 0, 6, 7, 8, 9};
  const int m2[] = {1, 0, 3, 0, 0, 0, 7, 0, 9};

  matrix_zero(m1, 3, 3);
  for (size_t i = 0; i < 9; i++) ASSERT_EQ(m2[i], m1[i]);

  float m3a[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  const float m3[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

  matrix_zero(m3a, 3, 3);
  for (size_t i = 0; i < 9; i++) ASSERT_EQ(m3a[i], m3[i]);

  int m4[] = {
      0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
  };
  const int m5[] = {
      0, 0, 0, 0, 0, 6, 7, 8, 0, 10, 11, 12,
  };

  matrix_zero(m4, 3, 4);
  for (size_t i = 0; i < 12; i++) ASSERT_EQ(m4[i], m5[i]);
}

////////////////
// Benchmarks //
////////////////

static const size_t BM_length_min = 8;
static const size_t BM_length_max = 10 << 5;

void BM_matrix_zero(benchmark::State &state) {
  const auto n = static_cast<size_t>(state.range(0));
  int *m = new int[n * n];

  while (state.KeepRunning()) {
    for (size_t i = 0; i < n * n; i++)
      m[i] = static_cast<int>(rand_r(&seed) % 25);

    matrix_zero(m, n, n);
    benchmark::DoNotOptimize(m[0]);
  }

  delete[] m;
}
BENCHMARK(BM_matrix_zero)->Range(BM_length_min, BM_length_max);

CTCI_MAIN();
