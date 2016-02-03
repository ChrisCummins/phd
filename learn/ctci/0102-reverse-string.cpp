/*
 * 1.2 Write a function void reverse(char *str) which reverses a
 * null-terminated string.
 */
#include "./ctci.h"

#include <cmath>
#include <cstdlib>
#include <limits>
#include <array>
#include <iostream>
#include <unordered_map>
#include <string>

void reverse1(char *str) {
    // First implementation. O(n) time.
    auto len = strlen(str) - 1;

    for (size_t i = 0; i <= len / 2; i++) {
        char c = str[i];
        str[i] = str[len - i];
        str[len - i] = c;
    }
}


TEST(Reverse, reverse1) {
    char test1[] = "abcdefg";
    char test2[] = "abcdefg ";

    reverse1(&test1[0]);
    reverse1(&test2[0]);

    ASSERT_STREQ("gfedcba", test1);
    ASSERT_STREQ(" gfedcba", test2);
}


static const size_t lengthMin = 8;
static const size_t lengthMax = 10 << 10;

void BM_reverse1(benchmark::State& state) {
    auto len = static_cast<size_t>(state.range_x());
    auto *t = new char[len];

    for (size_t i = 0; i < len; i++)
        t[i] = arc4random() % std::numeric_limits<char>::max();

    while (state.KeepRunning()) {
        reverse1(t);
        benchmark::DoNotOptimize(t);
    }

    delete[] t;
}
BENCHMARK(BM_reverse1)->Range(lengthMin, lengthMax);

CTCI_MAIN();
