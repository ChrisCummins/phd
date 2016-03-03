/*
 * Given a string s and a number N, find the longest substring of s
 * with maximum N unique characters.
 */
#include <limits>
#include <algorithm>
#include <iostream>
#include <string>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpadded"
#pragma GCC diagnostic ignored "-Wundef"
#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#pragma GCC diagnostic pop


std::string longest_substr(const std::string& s, const size_t n) {
  if (!n)  // do nothing if there's nothing to do
    return "";
  else if (n >= s.length())  // do nothing if we can
    return s;

  size_t start = 0, maxlen = 1, len = 0;

  for (size_t i = 0; i < s.length() - maxlen; i++) {
    std::bitset<std::numeric_limits<char>::max()> uniq{0};
    size_t j = i, count = 0;

    for ( ; j < s.length() && count <= n; j++) {
      if (!uniq[static_cast<size_t>(s[j])])
        ++count;
      uniq[static_cast<size_t>(s[j])] = true;
    }

    if ((len = j - i) > maxlen) {
      start = i;
      maxlen = len;
    }
  }

  return s.substr(start, maxlen);
}


///////////
// Tests //
///////////

TEST(basic, longest_substr) {
  ASSERT_EQ("ddddd", longest_substr("abcddddd", 1));
  ASSERT_EQ("cddddd", longest_substr("abcddddd", 2));
  ASSERT_EQ("cccddddd", longest_substr("abcccddddd", 2));
  ASSERT_EQ("bcccddddd", longest_substr("abcccddddd", 3));
  ASSERT_EQ("abcccddddd", longest_substr("abcccddddd", 10));
  ASSERT_EQ("", longest_substr("abcddddd", 0));
}


////////////////
// Benchmarks //
////////////////

static const size_t BM_length_min = 8;
static const size_t BM_length_max = 10 << 10;

void BM_baseline(benchmark::State& state) {
  std::string t(static_cast<size_t>(state.range_x()), 'a');

  while (state.KeepRunning()) {
    for (auto &c : t)  // generate string
      c = arc4random() % std::numeric_limits<char>::max();
    benchmark::DoNotOptimize(t.data());
  }
}
BENCHMARK(BM_baseline)->Range(BM_length_min, BM_length_max);

void BM_longest_substr(benchmark::State& state) {
  std::string t(static_cast<size_t>(state.range_x()), 'a');

  while (state.KeepRunning()) {
    for (auto &c : t)  // generate string
      c = arc4random() % std::numeric_limits<char>::max();

    longest_substr(t, (arc4random() % t.length()) / 2);
    benchmark::DoNotOptimize(t.data());
  }
}
BENCHMARK(BM_longest_substr)->Range(BM_length_min, BM_length_max);


int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  const auto ret = RUN_ALL_TESTS();
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  return ret;
}
