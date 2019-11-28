#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated"
#pragma GCC diagnostic ignored "-Wmissing-noreturn"
#pragma GCC diagnostic ignored "-Wpadded"
#pragma GCC diagnostic ignored "-Wundef"
#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

//
// Recursive fibonacci.
//
//   Time: O(2^n * T::operator+)
//   Space: O(n + n * T::operator+)
//
template <typename T>
T fib(const T& n) {
  static_assert(std::is_integral<T>::value, "error");

  if (!n) {
    return 0;
  } else if (n == 1) {
    return 1;
  } else {
    // Support negative parameters:
    T left, right;
    if (n < 0) {
      left = fib(n + 2);
      right = fib(n + 1);
    } else {
      left = fib(n - 1);
      right = fib(n - 2);
    }

    T res{left + right};
    if (res < left || res < right)
      throw std::runtime_error("return type overflow");
    else
      return res;
  }
}

//
// Iterative fibonacci.
//
//   Time: O(n * T::operator+)
//   Space: O(1) space
//
template <typename T>
T fib_iter(T n) {
  static_assert(std::is_integral<T>::value, "error");

  T x{0}, y{1}, z{0};

  if (n == 0)
    return 0;
  else if (n == 1)
    return 1;

  while (--n) {
    z = x + y;
    x = y;
    y = z;
  }

  return z;
}

//
// Fibonacci using memoisation. Implemented using hash table for
// lookups. Worst case performance equal to fib_iter(). Best case of
// O(1) time for lookup table hit.
//
template <typename T>
T fib_mem(const T& n) {
  static_assert(std::is_integral<T>::value, "error");
  static std::unordered_map<T, T> lookup_table;

  auto it = lookup_table.find(n);
  if (it == lookup_table.end()) {
    // Compute value:
    T res{fib_iter(n)};
    lookup_table.emplace(n, res);
    return res;
  } else {
    // Fetch from lookup table.
    return (*it).second;
  }
}

//
// Specialisation for int memoisation. Using a flat array instead of a
// hash table, since we can bound the valid sizes of .
//
int fib_mem(const int& n) {
  const size_t lookup_table_size = 47;
  const auto max_idx = static_cast<int>(lookup_table_size) - 1;

  static int lookup_table[lookup_table_size] = {};

  if (n < -max_idx && n > max_idx) throw std::out_of_range("lookup_table");

  const auto idx = abs(n);
  if (!lookup_table[idx]) lookup_table[idx] = fib_iter(n);

  return lookup_table[idx];
}

//
// Compile time fibonacci.
//
template <typename Container, unsigned int n>
class _fib {
 public:
  enum { val = _fib<Container, n - 1>::val + _fib<Container, n - 2>::val };
  static void compute(Container& result) {
    result[n] = val;
    _fib<Container, n - 1>::compute(result);
  }
};

// Partial specialization of _fib for 0.
template <typename Container>
class _fib<Container, 0> {
 public:
  enum { val };
  static void compute(Container& result) { result[0] = val; }
};

// Partial specialization of _fib for 1.
template <typename Container>
class _fib<Container, 1> {
 public:
  enum { val = 1 };
  static void compute(Container& result) {
    result[1] = val;
    _fib<Container, 0>::compute(result);
  }
};

//
// Computation of fibonacci sequence at compile time.
//
template <typename T, unsigned int n>
constexpr auto compile_time_fib() {
  static_assert(n > 0, "error");

  auto result = std::array<T, n>();
  _fib<decltype(result), n>::compute(result);
  return result;
}

///////////
// Tests //
///////////

TEST(fib, basic) {
  ASSERT_EQ(0, fib(0));
  ASSERT_EQ(1, fib(1));
  ASSERT_EQ(1, fib(2));
  ASSERT_EQ(2, fib(3));
  ASSERT_EQ(3, fib(4));
  ASSERT_EQ(5u, fib(5u));
  ASSERT_EQ(55u, fib(10u));
}

TEST(fib, negative) {
  ASSERT_EQ(1, fib(-1));
  ASSERT_EQ(1, fib(-2));
  ASSERT_EQ(2, fib(-3));
  ASSERT_EQ(3, fib(-4));
  ASSERT_EQ(5, fib(-5));
  ASSERT_EQ(8, fib(-6));
}

TEST(fib_mem, basic) {
  ASSERT_EQ(0, fib_mem(0));
  ASSERT_EQ(1, fib_mem(1));
  ASSERT_EQ(1, fib_mem(2));
  ASSERT_EQ(2, fib_mem(3));
  ASSERT_EQ(3, fib_mem(4));
  ASSERT_EQ(5, fib_mem(5));
  ASSERT_EQ(55, fib_mem(10));
  // table hits:
  ASSERT_EQ(0, fib_mem(0));
  ASSERT_EQ(1, fib_mem(1));
  ASSERT_EQ(1, fib_mem(2));
  ASSERT_EQ(2, fib_mem(3));
  ASSERT_EQ(3, fib_mem(4));
  ASSERT_EQ(5, fib_mem(5));
  ASSERT_EQ(55, fib_mem(10));
}

TEST(fib_mem, negative) {
  ASSERT_EQ(1, fib_mem(-1));
  ASSERT_EQ(1, fib_mem(-2));
  ASSERT_EQ(2, fib_mem(-3));
  ASSERT_EQ(3, fib_mem(-4));
  ASSERT_EQ(5, fib_mem(-5));
}

TEST(fib_mem, unsigned) {
  ASSERT_EQ(0u, fib_mem(0u));
  ASSERT_EQ(1u, fib_mem(1u));
  ASSERT_EQ(1u, fib_mem(2u));
  ASSERT_EQ(2u, fib_mem(3u));
  ASSERT_EQ(3u, fib_mem(4u));
  ASSERT_EQ(5u, fib_mem(5u));
  ASSERT_EQ(55u, fib_mem(10u));
  // table hits:
  ASSERT_EQ(0u, fib_mem(0u));
  ASSERT_EQ(1u, fib_mem(1u));
  ASSERT_EQ(1u, fib_mem(2u));
  ASSERT_EQ(2u, fib_mem(3u));
  ASSERT_EQ(3u, fib_mem(4u));
  ASSERT_EQ(5u, fib_mem(5u));
  ASSERT_EQ(55u, fib_mem(10u));
}

TEST(fib_iter, basic) {
  ASSERT_EQ(0, fib_iter(0));
  ASSERT_EQ(1, fib_iter(1));
  ASSERT_EQ(1, fib_iter(2));
  ASSERT_EQ(2, fib_iter(3));
  ASSERT_EQ(3, fib_iter(4));
  ASSERT_EQ(5, fib_iter(5));

  ASSERT_EQ(55u, fib_iter(10u));
}

TEST(fib_compile_time, basic) {
  auto seq = compile_time_fib<int, 11>();

  ASSERT_EQ(0, seq[0]);
  ASSERT_EQ(1, seq[1]);
  ASSERT_EQ(1, seq[2]);
  ASSERT_EQ(2, seq[3]);
  ASSERT_EQ(3, seq[4]);
  ASSERT_EQ(5, seq[5]);
  ASSERT_EQ(55, seq[10]);
}

////////////////
// Benchmarks //
////////////////

void BM_fib(benchmark::State& state) {
  const auto n = state.range(0);

  while (state.KeepRunning()) {
    auto ret = fib(n);
    benchmark::DoNotOptimize(ret);
  }
}
BENCHMARK(BM_fib)->Range(1, 25);

void BM_fib_mem(benchmark::State& state) {
  const auto n = state.range(0);

  while (state.KeepRunning()) {
    auto ret = fib_mem(n);
    benchmark::DoNotOptimize(ret);
  }
}
BENCHMARK(BM_fib_mem)->Range(1, 25);

void BM_fib_mem_unsigned(benchmark::State& state) {
  const auto n = static_cast<unsigned>(state.range(0));

  while (state.KeepRunning()) {
    auto ret = fib_mem(n);
    benchmark::DoNotOptimize(ret);
  }
}
BENCHMARK(BM_fib_mem_unsigned)->Range(1, 25);

void BM_fib_iter(benchmark::State& state) {
  const auto n = state.range(0);

  while (state.KeepRunning()) {
    auto ret = fib_iter(n);
    benchmark::DoNotOptimize(ret);
  }
}
BENCHMARK(BM_fib_iter)->Range(1, 25);

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  const auto ret = RUN_ALL_TESTS();
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  return ret;
}
