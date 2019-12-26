#include <stdlib.h>

#include <algorithm>
#include <array>
#include <ustl/algorithm>
#include <ustl/array>
#include <ustl/vector>
#include <vector>

#include "labm8/cpp/test.h"

static unsigned int seed = 0xCEC;

static const size_t sort_size_min = 8;
// FIXME: Change to a larger maximum sort size, and increase the
// number of increments:
static const size_t sort_size_max = 8 << 10;

static void std_algorithm_sort_int(benchmark::State& state) {
  std::vector<int> v(static_cast<size_t>(state.range(0)));

  while (state.KeepRunning()) {
    for (auto& i : v) i = static_cast<int>(rand_r(&seed));

    std::sort(v.begin(), v.end());
    benchmark::DoNotOptimize(v.data());
  }
}
BENCHMARK(std_algorithm_sort_int)->Range(sort_size_min, sort_size_max);

static void ustl_algorithm_sort_int(benchmark::State& state) {
  ustl::vector<int> v(static_cast<size_t>(state.range(0)));

  while (state.KeepRunning()) {
    for (auto& i : v) i = static_cast<int>(rand_r(&seed));

    ustl::sort(v.begin(), v.end());
    benchmark::DoNotOptimize(v.data());
  }
}
BENCHMARK(ustl_algorithm_sort_int)->Range(sort_size_min, sort_size_max);

TEST_MAIN();
