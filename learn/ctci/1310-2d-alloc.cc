/*
 * Write a function in C called my2DAlloc which allocates a
 * two-dimensional array. Minimize the number of calls to malloc and
 * make sure that the memory is accessible by the notation arr[i][j].
 */
#include "./ctci.h"

#include <cstdlib>

//
// IMPLEMENTATION NOTE: The challenge requires a C implementation, so
// my solutions use C-style casts, rather than reinterpretive_casts.
//
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wcast-align"

//
// First implementation. Allocate an array of pointers, and then
// allocate each row. Requires n+1 calls to malloc.
//
// O(n) time, O(1) space (in the sense that the memory overhead is
// determined solely by nrows and rowsize).
//
void** my2DAlloc1(size_t nrows, size_t rowsize) {
  void** rows = (void**)malloc(nrows * sizeof(void*));  // NOLINT

  while (nrows--) rows[nrows] = malloc(rowsize);

  return rows;
}

void my2DFree1(void** matrix, size_t nrows) {
  for (size_t i = 0; i < nrows; i++) free(matrix[i]);
  free(matrix);
}

//
// Second implementation. Allocate two arrays, one to hold pointer
// indexes, and another to store elements in a contiguous
// block. Requires 2 calls to malloc. Note that this is specialized to
// int type.
//
// O(n) time, O(1) space (in the sense that the memory overhead is
// determined solely by nrows and rowsize).
//
int** my2DAlloc2(size_t nrows, size_t ncols) {
  assert(ncols);

  int** rows = (int**)malloc(nrows * sizeof(int*));       // NOLINT
  int* data = (int*)malloc(nrows * ncols * sizeof(int));  // NOLINT

  while (nrows--) rows[nrows] = &data[nrows * ncols];

  return rows;  // NOLINT
}

void my2DFree2(int** matrix) {
  free(matrix[0]);
  free(matrix);
}

//
// Third implementation. Allocate as a single contiguous block of
// memory.
//
// O(n) time, O(1) space (in the sense that the memory overhead is
// determined solely by nrows and rowsize).
//
int** my2DAlloc3(size_t nrows, size_t ncols) {
  const size_t header_s = nrows * sizeof(int*);
  const size_t payload_s = nrows * ncols * sizeof(int);
  const size_t row_s = ncols * sizeof(int);

  int** data = (int**)malloc(header_s + payload_s);  // NOLINT

  while (nrows--)
    data[nrows] = (int*)&((char*)data)[header_s + nrows * row_s];  // NOLINT

  return data;
}

void my2DFree3(int** matrix) { free(matrix); }

///////////
// Tests //
///////////

TEST(my2DAlloc1, tests) {
  int** matrix = (int**)my2DAlloc1(3, 4 * sizeof(int));  // NOLINT

  matrix[0][0] = 0;
  matrix[0][1] = 1;
  matrix[0][2] = 2;
  matrix[0][3] = 3;

  matrix[1][0] = 4;
  matrix[1][1] = 5;
  matrix[1][2] = 6;
  matrix[1][3] = 7;

  matrix[2][0] = 8;
  matrix[2][1] = 9;
  matrix[2][2] = 10;
  matrix[2][3] = 11;

  ASSERT_EQ(matrix[0][0], 0);
  ASSERT_EQ(matrix[0][1], 1);
  ASSERT_EQ(matrix[0][2], 2);
  ASSERT_EQ(matrix[0][3], 3);

  ASSERT_EQ(matrix[1][0], 4);
  ASSERT_EQ(matrix[1][1], 5);
  ASSERT_EQ(matrix[1][2], 6);
  ASSERT_EQ(matrix[1][3], 7);

  ASSERT_EQ(matrix[2][0], 8);
  ASSERT_EQ(matrix[2][1], 9);
  ASSERT_EQ(matrix[2][2], 10);
  ASSERT_EQ(matrix[2][3], 11);

  my2DFree1((void**)matrix, 4 * sizeof(int));  // NOLINT
}

TEST(my2DAlloc2, tests) {
  int** matrix = my2DAlloc2(3, 4);

  matrix[0][0] = 0;
  matrix[0][1] = 1;
  matrix[0][2] = 2;
  matrix[0][3] = 3;

  matrix[1][0] = 4;
  matrix[1][1] = 5;
  matrix[1][2] = 6;
  matrix[1][3] = 7;

  matrix[2][0] = 8;
  matrix[2][1] = 9;
  matrix[2][2] = 10;
  matrix[2][3] = 11;

  ASSERT_EQ(matrix[0][0], 0);
  ASSERT_EQ(matrix[0][1], 1);
  ASSERT_EQ(matrix[0][2], 2);
  ASSERT_EQ(matrix[0][3], 3);

  ASSERT_EQ(matrix[1][0], 4);
  ASSERT_EQ(matrix[1][1], 5);
  ASSERT_EQ(matrix[1][2], 6);
  ASSERT_EQ(matrix[1][3], 7);

  ASSERT_EQ(matrix[2][0], 8);
  ASSERT_EQ(matrix[2][1], 9);
  ASSERT_EQ(matrix[2][2], 10);
  ASSERT_EQ(matrix[2][3], 11);

  my2DFree2(matrix);
}

TEST(my2DAlloc3, tests) {
  int** matrix = my2DAlloc3(3, 4);

  matrix[0][0] = 0;
  matrix[0][1] = 1;
  matrix[0][2] = 2;
  matrix[0][3] = 3;

  matrix[1][0] = 4;
  matrix[1][1] = 5;
  matrix[1][2] = 6;
  matrix[1][3] = 7;

  matrix[2][0] = 8;
  matrix[2][1] = 9;
  matrix[2][2] = 10;
  matrix[2][3] = 11;

  ASSERT_EQ(matrix[0][0], 0);
  ASSERT_EQ(matrix[0][1], 1);
  ASSERT_EQ(matrix[0][2], 2);
  ASSERT_EQ(matrix[0][3], 3);

  ASSERT_EQ(matrix[1][0], 4);
  ASSERT_EQ(matrix[1][1], 5);
  ASSERT_EQ(matrix[1][2], 6);
  ASSERT_EQ(matrix[1][3], 7);

  ASSERT_EQ(matrix[2][0], 8);
  ASSERT_EQ(matrix[2][1], 9);
  ASSERT_EQ(matrix[2][2], 10);
  ASSERT_EQ(matrix[2][3], 11);

  my2DFree3(matrix);
}

////////////////
// Benchmarks //
////////////////

static const size_t BM_size_min = 8;
static const size_t BM_size_max = 10 << 10;

void BM_my2DAlloc1(benchmark::State& state) {
  const size_t size = static_cast<size_t>(state.range(0));
  assert(size);

  while (state.KeepRunning()) {
    int** matrix = (int**)my2DAlloc1(size, size * sizeof(int));  // NOLINT
    benchmark::DoNotOptimize(matrix[0][0]);
    my2DFree1((void**)matrix, size);  // NOLINT
  }
}
BENCHMARK(BM_my2DAlloc1)->Range(BM_size_min, BM_size_max);

void BM_my2DAlloc2(benchmark::State& state) {
  const size_t size = static_cast<size_t>(state.range(0));
  assert(size);

  while (state.KeepRunning()) {
    int** matrix = my2DAlloc2(size, size);
    benchmark::DoNotOptimize(matrix[0][0]);
    my2DFree2(matrix);
  }
}
BENCHMARK(BM_my2DAlloc2)->Range(BM_size_min, BM_size_max);

void BM_my2DAlloc3(benchmark::State& state) {
  const size_t size = static_cast<size_t>(state.range(0));
  assert(size);

  while (state.KeepRunning()) {
    int** matrix = my2DAlloc3(size, size);
    benchmark::DoNotOptimize(matrix[0][0]);
    my2DFree3(matrix);
  }
}
BENCHMARK(BM_my2DAlloc3)->Range(BM_size_min, BM_size_max);

#pragma GCC diagnostic pop

CTCI_MAIN();
