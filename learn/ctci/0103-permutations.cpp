/*
 * Given two strings, write a method to decide if one is a permutation
 * of another.
 */

#include <cmath>
#include <cstdlib>
#include <limits>
#include <array>
#include <iostream>
#include <unordered_map>
#include <string>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpadded"
#pragma GCC diagnostic ignored "-Wundef"
#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

bool isPermutation1(std::string &a, std::string &b) {
    // Optimization - permutations are the same length.
    if (a.size() != b.size())
        return false;

    std::array<int, std::numeric_limits<char>::max()> af = {}, bf = {};

    // Create char-frequency tables:
    for (auto &c : a)
        af[static_cast<size_t>(c)] += 1;
    for (auto &c : b)
        bf[static_cast<size_t>(c)] += 1;

    // Check char-frequency equivalence.
    for (size_t i = 0; i < af.size(); i++) {
        if (af[i] != bf[i])
            return false;
    }

    return true;
}

// Unit tests

TEST(Permutation, isPermutation1) {
    std::string a1("abcde"), a2("efgh");
    std::string b1("abcde"), b2("edcba");

    ASSERT_FALSE(isPermutation1(a1, a2));
    ASSERT_TRUE(isPermutation1(b1, b2));
}

// Benchmarks

static unsigned int seed = 0xcec;
static const size_t lengthMin = 8;
static const size_t lengthMax = 10 << 10;

void BM_baseline(benchmark::State& state) {
    auto len = static_cast<size_t>(state.range_x());
    std::string t1(len, 'a'), t2(len, 'a');

    while (state.KeepRunning()) {
        for (auto &c : t1)
            c = rand_r(&seed) % std::numeric_limits<char>::max();
        for (auto &c : t2)
            c = rand_r(&seed) % std::numeric_limits<char>::max();

        benchmark::DoNotOptimize(t1.data());
        benchmark::DoNotOptimize(t2.data());
    }
}
BENCHMARK(BM_baseline)->Range(lengthMin, lengthMax);

void BM_isPermutation1(benchmark::State& state) {
    auto len = static_cast<size_t>(state.range_x());
    std::string t1(len, 'a'), t2(len, 'a');

    while (state.KeepRunning()) {
        for (auto &c : t1)
            c = rand_r(&seed) % std::numeric_limits<char>::max();
        for (auto &c : t2)
            c = rand_r(&seed) % std::numeric_limits<char>::max();

        auto ret = isPermutation1(t1, t2);
        benchmark::DoNotOptimize(ret);
    }
}
BENCHMARK(BM_isPermutation1)->Range(lengthMin, lengthMax);

int main(int argc, char **argv) {
    // Run unit tests:
    testing::InitGoogleTest(&argc, argv);
    const auto ret = RUN_ALL_TESTS();

    // Run benchmarks:
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();

    return ret;
}
