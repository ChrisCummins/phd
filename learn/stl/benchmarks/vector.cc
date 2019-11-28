#include "./benchmarks.h"

#include <ustl/vector>
#include <vector>

static unsigned int seed = 0xCEC;

static void std_vector_push_back_int(benchmark::State& state) {
  std::vector<int> v(0);

  while (state.KeepRunning()) {
    v.push_back(static_cast<int>(rand_r(&seed)));
    benchmark::DoNotOptimize(v.data());
  }
}
BENCHMARK(std_vector_push_back_int);

static void ustl_vector_push_back_int(benchmark::State& state) {
  ustl::vector<int> v(0);

  while (state.KeepRunning()) {
    v.push_back(static_cast<int>(rand_r(&seed)));
    benchmark::DoNotOptimize(v.data());
  }
}
BENCHMARK(ustl_vector_push_back_int);
