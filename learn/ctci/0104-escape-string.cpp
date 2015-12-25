/*
 * Write a method to replace all spaces in a string with '%20'.
 */

#include <algorithm>
#include <limits>
#include <cstdlib>
#include <iostream>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpadded"
#pragma GCC diagnostic ignored "-Wundef"
#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

void escapeSpace1(char *s, const size_t len) {
    std::vector<size_t> idxs;

    // Create a list of space indices:
    for (size_t i = 0; i < len; i++)
        if (s[i] == ' ')
            idxs.push_back(i);
    std::reverse(idxs.begin(), idxs.end());

    for (size_t i = 0; i < idxs.size(); i++) {
        auto idx = idxs[i];

        for (size_t j = len - 1 + 2 * i; j > idx; j--)
            s[j + 2] = s[j];

        s[idx] = '%';
        s[idx + 1] = '2';
        s[idx + 2] = '0';
    }
}

// Unit tests

TEST(Permutation, escapeSpace1) {
    char a[] = "abcde";
    escapeSpace1(a, 5);
    ASSERT_STREQ("abcde", a);

    char b[] = "abc de  ";
    escapeSpace1(b, 6);
    ASSERT_STREQ("abc%20de", b);

    char c[] = "a bc de    ";
    escapeSpace1(c, 7);
    ASSERT_STREQ("a%20bc%20de", c);
}

// Benchmarks

static const size_t lengthMin = 8;
static const size_t lengthMax = 10 << 10;

void populateString(char *const t, const size_t strlen, size_t *len) {
    for (size_t i = 0; i < strlen; i++) {
        if (!arc4random() % 4)
            t[i] = ' ';
        else
            t[i] = static_cast<char>(arc4random()
                                     % std::numeric_limits<char>::max());
    }

    *len = strlen;
    for (size_t i = 0; i < strlen; i++)
        if (t[i] == ' ')
            *len -= 2;
}

void BM_baseline(benchmark::State& state) {
    const auto strlen = static_cast<size_t>(state.range_x());
    char *const t = new char[strlen];
    size_t len = 0;


    while (state.KeepRunning()) {
        populateString(t, strlen, &len);
        benchmark::DoNotOptimize(*t);
    }

    delete[] t;
}
BENCHMARK(BM_baseline)->Range(lengthMin, lengthMax);

void BM_escapeSpace1(benchmark::State& state) {
    const auto strlen = static_cast<size_t>(state.range_x());
    char *const t = new char[strlen];
    size_t len = 0;


    while (state.KeepRunning()) {
        populateString(t, strlen, &len);
        escapeSpace1(t, len);
        benchmark::DoNotOptimize(*t);
    }

    delete[] t;
}
BENCHMARK(BM_escapeSpace1)->Range(lengthMin, lengthMax);


int main(int argc, char **argv) {
    // Run unit tests:
    testing::InitGoogleTest(&argc, argv);
    const auto ret = RUN_ALL_TESTS();

    // Run benchmarks:
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();

    return ret;
}
