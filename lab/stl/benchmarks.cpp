#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpadded"
#pragma GCC diagnostic ignored "-Wundef"
#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

#include <algorithm>
#include <ustl/algorithm>

#include <array>
#include <ustl/array>

#include <vector>
#include <ustl/vector>


// Algorithm benchmarks


static const size_t sort_size_min = 8;
// FIXME: Change to a larger maximum sort size, and increase the
// number of increments:
static const size_t sort_size_max = 8 << 10;

static void std_sort_int(benchmark::State& state) {
  std::vector<int> v(static_cast<size_t>(state.range_x()));

  while (state.KeepRunning()) {
    for (auto& i : v)
      i = static_cast<int>(arc4random());

    std::sort(v.begin(), v.end());
    benchmark::DoNotOptimize(v.data());
  }
}
BENCHMARK(std_sort_int)->Range(sort_size_min, sort_size_max);

static void ustl_sort_int(benchmark::State& state) {
  ustl::vector<int> v(static_cast<size_t>(state.range_x()));

  while (state.KeepRunning()) {
    for (auto& i : v)
      i = static_cast<int>(arc4random());

    ustl::sort(v.begin(), v.end());
    benchmark::DoNotOptimize(v.data());
  }
}
BENCHMARK(ustl_sort_int)->Range(sort_size_min, sort_size_max);


// Array tests


// Vector tests


static void std_push_back_int(benchmark::State& state) {
  std::vector<int> v(0);

  while (state.KeepRunning()) {
    v.push_back(static_cast<int>(arc4random()));
    benchmark::DoNotOptimize(v.data());
  }
}
BENCHMARK(std_push_back_int);

static void ustl_push_back_int(benchmark::State& state) {
  ustl::vector<int> v(0);

  while (state.KeepRunning()) {
    v.push_back(static_cast<int>(arc4random()));
    benchmark::DoNotOptimize(v.data());
  }
}
BENCHMARK(ustl_push_back_int);


int main(int argc, char **argv) {
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}
