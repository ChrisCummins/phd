/*
 * Write a function to swap a number in place (that is, without
 * temporary variables).
 */
#include "./ctci.h"

//
// Requires a temporary, not a solution.
//
template <typename T>
void num_swap(T *const x, T *const y) {
  T tmp = *x;

  *x = *y;
  *y = tmp;
}

//
// Solution: XOR trick.
//
// Lets run through an example:
//
//   num x: 1011
//   num y: 1100
//
// 1. XOR x and y, writing result to x:
//
//       x 1011
//       y 1100
//   XOR   0111
//
// 2. XOR x and y, writing the result to y:
//
//       x 0111
//       y 1100
//   XOR   1011
//
// 3. XOR x and y, writing the result to x:
//
//       x 0111
//       y 1011
//   XOR   1100
//
// Visually:
//
//    1011 --> 1100 -----+
//      |       ^        |
//      v       |        v
//    1100 --> 0111 --> 1011
//
template <typename T>
void inplace_num_swap(T *const x, T *const y) {
  *x ^= *y;
  *y ^= *x;
  *x ^= *y;
}

///////////
// Tests //
///////////

TEST(NumSwap, inplace_num_swap_int) {
  int x = 5, y = 10;

  inplace_num_swap(&x, &y);

  ASSERT_EQ(10, x);
  ASSERT_EQ(5, y);

  inplace_num_swap(&x, &y);

  ASSERT_EQ(5, x);
  ASSERT_EQ(10, y);
}

TEST(NumSwap, inplace_num_swap_char) {
  char x = 'a', y = 'b';

  inplace_num_swap(&x, &y);

  ASSERT_EQ('b', x);
  ASSERT_EQ('a', y);

  inplace_num_swap(&x, &y);

  ASSERT_EQ('a', x);
  ASSERT_EQ('b', y);
}

////////////////
// Benchmarks //
////////////////

void BM_num_swap(benchmark::State &state) {
  int x = 5, y = 10;

  while (state.KeepRunning()) {
    num_swap(&x, &y);
    benchmark::DoNotOptimize(x);
    benchmark::DoNotOptimize(y);
  }
}
BENCHMARK(BM_num_swap);

void BM_inplace_num_swap_int(benchmark::State &state) {
  int x = 5, y = 10;

  while (state.KeepRunning()) {
    inplace_num_swap(&x, &y);
    benchmark::DoNotOptimize(x);
    benchmark::DoNotOptimize(y);
  }
}
BENCHMARK(BM_inplace_num_swap_int);

void BM_inplace_num_swap_char(benchmark::State &state) {
  int x = 'a', y = 'b';

  while (state.KeepRunning()) {
    inplace_num_swap(&x, &y);
    benchmark::DoNotOptimize(x);
    benchmark::DoNotOptimize(y);
  }
}
BENCHMARK(BM_inplace_num_swap_char);

CTCI_MAIN();
