#include <stdlib.h>
#include <algorithm>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpadded"
#include <benchmark/benchmark.h>
#pragma GCC diagnostic pop

static unsigned int seed = 0xCEC;

static void lambda_loop(benchmark::State& state) {
  static const size_t n = 1000;
  std::vector<int> v(n);
  for (auto& i : v) i = static_cast<int>(rand_r(&seed));
  auto sum = 0;

  auto op = [&](const int& x) { sum += x; };
  while (state.KeepRunning()) {
    std::for_each(v.begin(), v.end(), op);
    benchmark::DoNotOptimize(sum);
  }
}
BENCHMARK(lambda_loop);

static void lambda_in_loop(benchmark::State& state) {
  static const size_t n = 1000;
  std::vector<int> v(n);
  for (auto& i : v) i = static_cast<int>(rand_r(&seed));
  auto sum = 0;

  while (state.KeepRunning()) {
    std::for_each(v.begin(), v.end(), [&](const int& x) { sum += x; });
    benchmark::DoNotOptimize(sum);
  }
}
BENCHMARK(lambda_in_loop);

static void lambda_val(benchmark::State& state) {
  static const size_t n = 1000;
  std::vector<int> v(n);
  for (auto& i : v) i = static_cast<int>(rand_r(&seed));

  const auto m = 2;
  while (state.KeepRunning()) {
    std::for_each(v.begin(), v.end(), [=](int& x) { x *= m; });
    benchmark::DoNotOptimize(v.data());
  }
}
BENCHMARK(lambda_val);

static void lambda_ref(benchmark::State& state) {
  static const size_t n = 1000;
  std::vector<int> v(n);
  for (auto& i : v) i = static_cast<int>(rand_r(&seed));

  const auto m = 2;
  while (state.KeepRunning()) {
    std::for_each(v.begin(), v.end(), [&](int& x) { x *= m; });
    benchmark::DoNotOptimize(v.data());
  }
}
BENCHMARK(lambda_ref);

BENCHMARK_MAIN();
