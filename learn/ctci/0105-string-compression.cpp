/*
 * Write a method to perform basic string compression using the counts
 * of repeated characters. For example, the string aabcccccaaa would
 * become a2b1c5a3. If the "compressed" string would not become
 * smaller than the original string, your method should return the
 * original string.
 */
#include "./ctci.h"

#include <string>
#include <iostream>

std::string compressString1(std::string &s) {
    // Output string.
    std::ostringstream o;

    // Current character, and character count.
    char c = '\0';
    int cc = 0;

    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == c) {
            cc++;
        } else {
            if (c != '\0')
                o << c << cc;
            c = s[i];
            cc = 1;
        }
    }

    if (c != '\0')
        o << c << cc;

    // Return the shortest string:
    std::string ostr = o.str();
    return ostr.size() < s.size() ? ostr : s;
}

// Unit tests

TEST(Permutation, compressString1) {
    auto in1 = std::string("aabcccccaaa");
    auto out1 = compressString1(in1);
    ASSERT_EQ(std::string("a2b1c5a3"), out1);

    auto in2 = std::string("abcde");
    auto out2 = compressString1(in2);
    ASSERT_EQ(std::string("abcde"), out2);
}

// Benchmarks

static unsigned int seed = 0xcec;
static const size_t lengthMin = 8;
static const size_t lengthMax = 10 << 10;

void BM_baseline(benchmark::State& state) {
    const auto strlen = static_cast<size_t>(state.range_x());
    std::string t;

     while (state.KeepRunning()) {
         for (size_t i = 0; i < strlen; i++)
             t += static_cast<char>(rand_r(&seed) % 5);

         benchmark::DoNotOptimize(t[0]);
     }
}
BENCHMARK(BM_baseline)->Range(lengthMin, lengthMax);

void BM_compressString1(benchmark::State& state) {
    const auto strlen = static_cast<size_t>(state.range_x());
    std::string t;

    while (state.KeepRunning()) {
        for (size_t i = 0; i < strlen; i++)
            t += static_cast<char>(rand_r(&seed) % 5);

        std::string o = compressString1(t);
        benchmark::DoNotOptimize(o[0]);
    }
}
BENCHMARK(BM_compressString1)->Range(lengthMin, lengthMax);

CTCI_MAIN();
