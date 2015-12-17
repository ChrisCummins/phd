// Playing around with memory.

#include <cstdlib>
#include <iostream>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpadded"
#include <benchmark/benchmark.h>
#pragma GCC diagnostic pop

#include "./memory.h"

static const size_t size_min = 8;
static const size_t size_max = 8 << 10;

static unsigned int seed = 0xcec;


// Increment all values in a container.
template<typename Container>
void incrementByReference(Container &c) {
    for (auto &v : c)
        v++;
}

template<typename Container>
void incrementByPointer(Container *c) {
    for (auto &v : *c)
        v++;
}

template<typename Container>
void incrementByValue(Container c) {
    for (auto &v : c)
        v++;
}

// Print all values in a container.
template<typename Container>
void print(Container &c) {
    for (auto &v : c)
        std::cout << v << " ";
    std::cout << std::endl;
}


static void VectorReference(benchmark::State& state) {
    std::vector<int> v(static_cast<size_t>(state.range_x()));
    for (auto &i : v)
        i = rand_r(&seed);

    while (state.KeepRunning()) {
        incrementByReference(v);
        benchmark::DoNotOptimize(v.data());
    }
}
BENCHMARK(VectorReference)->Range(size_min, size_max);

static void VectorPointer(benchmark::State& state) {
    std::vector<int> v(static_cast<size_t>(state.range_x()));
    for (auto &i : v)
        i = rand_r(&seed);

    while (state.KeepRunning()) {
        incrementByPointer(&v);
        benchmark::DoNotOptimize(v.data());
    }
}
BENCHMARK(VectorPointer)->Range(size_min, size_max);

static void VectorValue(benchmark::State& state) {
    std::vector<int> v(static_cast<size_t>(state.range_x()));
    for (auto &i : v)
        i = rand_r(&seed);

    while (state.KeepRunning()) {
        incrementByValue(v);
        benchmark::DoNotOptimize(v.data());
    }
}
BENCHMARK(VectorValue)->Range(size_min, size_max);

BENCHMARK_MAIN();
