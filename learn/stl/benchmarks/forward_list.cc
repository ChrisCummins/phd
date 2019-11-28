#include "./benchmarks.h"

#include <forward_list>
#include <ustl/forward_list>

static unsigned int seed = 0xCEC;

static const size_t sort_size_min = 8;
// FIXME: Change to a larger maximum sort size, and increase the
// number of increments:
static const size_t sort_size_max = 8 << 10;

static void std_forward_list_sort_int(benchmark::State& state) {
  std::forward_list<int> list(static_cast<size_t>(state.range(0)));

  while (state.KeepRunning()) {
    for (auto& i : list) i = static_cast<int>(rand_r(&seed));

    list.sort();
    benchmark::DoNotOptimize(list.front());
  }
}
BENCHMARK(std_forward_list_sort_int)->Range(sort_size_min, sort_size_max);

static void ustl_forward_list_sort_int(benchmark::State& state) {
  ustl::forward_list<int> list(static_cast<size_t>(state.range(0)));

  while (state.KeepRunning()) {
    for (auto& i : list) i = static_cast<int>(rand_r(&seed));

    list.sort();
    benchmark::DoNotOptimize(list.front());
  }
}
BENCHMARK(ustl_forward_list_sort_int)->Range(sort_size_min, sort_size_max);
