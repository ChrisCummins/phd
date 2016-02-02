#include "./benchmarks.h"

#include <vector>
#include <ustl/vector>


static void std_vector_push_back_int(benchmark::State& state) {
  std::vector<int> v(0);

  while (state.KeepRunning()) {
    v.push_back(static_cast<int>(arc4random()));
    benchmark::DoNotOptimize(v.data());
  }
}
BENCHMARK(std_vector_push_back_int);

static void ustl_vector_push_back_int(benchmark::State& state) {
  ustl::vector<int> v(0);

  while (state.KeepRunning()) {
    v.push_back(static_cast<int>(arc4random()));
    benchmark::DoNotOptimize(v.data());
  }
}
BENCHMARK(ustl_vector_push_back_int);
