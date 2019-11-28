// Average Calculation
//
// Given a list of integers, write a function calculate their average.
//
// Calculate running average after each integer.
//
// Find (and fix) possible boundary cases that might cause exceptions.

#include <cstdint>

#include <stdlib.h>

#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

static unsigned int seed = 0xCEC;

float averageIntList1(int32_t *l, const size_t n) {
  /*
   * First, naive implementation.
   *
   * O(n) running time.
   * Accumulator may overflow.
   * Bug if argument 'n' is not equal to length of int array 'l'.
   * Not a generic implementation. Could use templates and iterators.
   * For large lists, could parallelise using a map reduce.
   * Note we *could* add a check to estimate whether the sum will
   * overflow, but would be unable to handle the situation (with
   * this algorithm).
   */
  if (n == 0 || l == nullptr) return 0;  // safety check

  int64_t sum = 0;
  for (size_t i = 0; i < n; i++) sum += l[i];

  return static_cast<float>(sum) / static_cast<float>(n);
}

float averageIntList2(int32_t *l, const size_t n) {
  /*
   * Second implementation, using a moving average.
   *
   * O(n) running time, but slower.
   *
   *
   *
   * Can also be map reduced, but can be parallelised to only a
   * coarser level.
   */
  if (n == 0 || l == nullptr) return 0;  // safety check

  float avg = l[0];
  size_t i = 1;
  while (i < n) {
    i++;
    avg -= avg / static_cast<float>(i);
    avg += l[i - 1] / static_cast<float>(i);
  }

  return avg;
}

// Unit tests

TEST(Correctness, averageIntList1) {
  int32_t t1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  ASSERT_FLOAT_EQ(5, averageIntList1(t1, 9));

  int32_t t2[] = {-3, -2, 0};
  ASSERT_FLOAT_EQ(-1.6666666f, averageIntList1(t2, 3));

  ASSERT_FLOAT_EQ(0, averageIntList1(nullptr, 0));
}

TEST(Correctness, averageIntList2) {
  int32_t t1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  ASSERT_FLOAT_EQ(averageIntList2(t1, 9), 5);

  int32_t t2[] = {-3, -2, 0};
  ASSERT_FLOAT_EQ(averageIntList2(t2, 3), -1.6666666f);

  ASSERT_FLOAT_EQ(averageIntList2(nullptr, 0), 0);
}

// Benchmarks

static const size_t lengthMin = 8;
static const size_t lengthMax = 10 << 10;

void BM_baseline(benchmark::State &state) {
  const auto n = static_cast<size_t>(state.range(0));
  auto *m = new int32_t[n];

  while (state.KeepRunning()) {
    for (size_t i = 0; i < n; i++) m[i] = static_cast<int32_t>(rand_r(&seed));

    benchmark::DoNotOptimize(*m);
  }

  delete[] m;
}
BENCHMARK(BM_baseline)->Range(lengthMin, lengthMax);

void BM_averageIntList1(benchmark::State &state) {
  const auto n = static_cast<size_t>(state.range(0));
  int *m = new int[n];

  while (state.KeepRunning()) {
    for (size_t i = 0; i < n; i++) m[i] = static_cast<int>(rand_r(&seed));

    averageIntList1(m, n);
    benchmark::DoNotOptimize(*m);
  }

  delete[] m;
}
BENCHMARK(BM_averageIntList1)->Range(lengthMin, lengthMax);

void BM_averageIntList2(benchmark::State &state) {
  const auto n = static_cast<size_t>(state.range(0));
  int *m = new int[n];

  while (state.KeepRunning()) {
    for (size_t i = 0; i < n; i++) m[i] = static_cast<int>(rand_r(&seed));

    averageIntList2(m, n);
    benchmark::DoNotOptimize(*m);
  }

  delete[] m;
}
BENCHMARK(BM_averageIntList2)->Range(lengthMin, lengthMax);

int main(int argc, char **argv) {
  // Run unit tests:
  testing::InitGoogleTest(&argc, argv);
  const auto ret = RUN_ALL_TESTS();

  // Run benchmarks:
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();

  return ret;
}
