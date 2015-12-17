/*
 * 1.1 Implement an algorithm to determine if a string has all unique
 * characters. What if you cannot use additional data structures?
 */

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

bool unique1(const std::string &s) {
    // First implementation. Uses a hash map to store whether a
    // character has already appeared before in a the string. O(n)
    // time.

    std::unordered_map<char, bool> map;

    for (auto &c : s) {
        if (map[c])
            return false;
        map[c] = true;
    }

    return true;
}

bool unique2(const std::string &s) {
    // Second implementation. Since the total number of possible
    // unique characters is pretty small (the size of a word), we can
    // replace the hash map with a array, and index into it by casting
    // character values to indexes.
    //
    // Requires a bit mask of size n, where n is the number of unique
    // character values. O(n) time.
    std::array<bool, std::numeric_limits<char>::max()> a = {};

    for (auto &c : s) {
        // Cast character to array index:
        const auto i = static_cast<size_t>(c);

        if (a[i])
            return false;

        a[i] = true;
    }

    // delete[] a;

    return true;
}

bool unique3(const std::string &s) {
    // Additional constraint: no extra data structures.
    //
    // No additional data structures. O(n^2) time.

    for (size_t i = 0; i < s.size() - 1; i++) {
        for (size_t j = 0; j < s.size(); j++) {
            if (i != j && s[i] == s[j])
                return false;
        }
    }

    return true;
}

// Unit tests

static const std::string isUnique("abcdefg ");
static const std::string notUnique("abcdefga");

TEST(Unique, unique1) {
    ASSERT_TRUE(unique1(isUnique));
    ASSERT_FALSE(unique1(notUnique));
}

TEST(Unique, unique2) {
    ASSERT_TRUE(unique2(isUnique));
    ASSERT_FALSE(unique2(notUnique));
}

TEST(Unique, unique3) {
    ASSERT_TRUE(unique3(isUnique));
    ASSERT_FALSE(unique3(notUnique));
}

// Benchmarks

static unsigned int seed = 0xcec;
static const size_t lengthMin = 8;
static const size_t lengthMax = 10 << 10;

void BM_unique1(benchmark::State& state) {
    std::string t(static_cast<size_t>(state.range_x()), 'a');

    while (state.KeepRunning()) {
        for (auto &c : t)
            c = rand_r(&seed) % std::numeric_limits<char>::max();

        unique1(t);
        benchmark::DoNotOptimize(t.data());
    }
}
BENCHMARK(BM_unique1)->Range(lengthMin, lengthMax);

void BM_unique2(benchmark::State& state) {
    std::string t(static_cast<size_t>(state.range_x()), 'a');

    while (state.KeepRunning()) {
        for (auto &c : t)
            c = rand_r(&seed) % std::numeric_limits<char>::max();

        unique2(t);
        benchmark::DoNotOptimize(t.data());
    }
}
BENCHMARK(BM_unique2)->Range(lengthMin, lengthMax);

void BM_unique3(benchmark::State& state) {
    std::string t(static_cast<size_t>(state.range_x()), 'a');

    while (state.KeepRunning()) {
        for (auto &c : t)
            c = rand_r(&seed) % std::numeric_limits<char>::max();

        unique3(t);
        benchmark::DoNotOptimize(t.data());
    }
}
BENCHMARK(BM_unique3)->Range(lengthMin, lengthMax);

int main(int argc, char **argv) {
    // Run unit tests:
    testing::InitGoogleTest(&argc, argv);
    const auto ret = RUN_ALL_TESTS();

    // Run benchmarks:
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();

    return ret;
}
