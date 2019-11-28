/*
 * Given a string s and a number N, find the longest substring of s
 * with maximum N unique characters.
 */
#include <stdlib.h>
#include <algorithm>
#include <array>
#include <bitset>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

static unsigned int seed = 0xCEC;

// My code
//
std::string maxnsubstr_chris(const std::string& s, const size_t n) {
  if (!n)  // do nothing if there's nothing to do
    return "";
  else if (n >= s.length())  // do nothing if we can
    return s;

  size_t start = 0, maxlen = 1, len = 0;

  for (size_t i = 0; i < s.length() - maxlen; i++) {
    std::bitset<std::numeric_limits<char>::max()> uniq{0};
    size_t j = i, count = 0;

    for (; j < s.length() && count <= n; j++) {
      if (!uniq[static_cast<size_t>(s[j])]) ++count;
      uniq[static_cast<size_t>(s[j])] = true;
    }

    if ((len = j - i) > maxlen) {
      start = i;
      maxlen = len;
    }
  }

  return s.substr(start, maxlen);
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wsign-compare"

// Adam's implementation: use a btree to store the "last" seen index
// of each character, and use that to calculate where to trim the
// start of the string to
//
std::string maxnsubstr_adam(std::string s, int N) {
  // optimised lookup table for most recent N characters
  std::map<char, int> most_recent_indices;

  int unique_count = 0;
  int i = 0, j = 0;
  int best_length = 0, best_j = 0;

  for (; i < s.length(); i++) {
    // if we're finding a new character, increase the number of seen uniques
    if (most_recent_indices.count(s[i]) == 0) {
      unique_count++;
    }
    // store i as the most recent index of the character O(N)
    most_recent_indices[s[i]] = i;
    // if we've added too many characters, correct using the
    // most recent indices to select a character to remove
    if (unique_count > N) {
      // prune the substring using he least recently seen O(N)
      auto min = most_recent_indices.begin();
      char min_char = min->first;
      int min_index = min->second;
      // erase it from the min list
      most_recent_indices.erase(min_char);
      // set the start of the new string as min_index + 1
      j = min_index + 1;
      // decrement the unique count
      unique_count--;
    }
    // get the size of the current substring
    int current_length = (i - j) + 1;
    // update the best length if we've improved it
    if (current_length > best_length) {
      best_length = current_length;
      best_j = j;
    }
  }
  return s.substr(best_j, best_length);
}

// Christophe's implementation: use occurence counts for each
// character to manually move the left hand pointer until one
// occurence count is zero, at which point we've managed to remove a
// unique character from the substring
//
std::string maxnsubstr_christophe(std::string s, int N) {
  // lookup table from characters -> # of occurences in current substring
  std::array<int, 128> occurences;
  occurences.fill(0);

  int unique_count = 0;
  int i = 0, j = 0;
  int best_length = 0, best_j = 0;

  for (i = 0; i < s.length(); i++) {
    // if we're finding a new character, increase the number of seen uniques
    if (occurences[s[i]] == 0) {
      unique_count++;
    }
    // increase the number of times we've seen this character
    occurences[s[i]]++;
    // correct for uniques if required by moving the "left" pointer along
    while (unique_count > N) {
      occurences[s[j]]--;
      if (occurences[s[j]] == 0) {
        unique_count--;
      }
      j++;
    }
    // get the size of the current substring
    int current_length = (i - j) + 1;
    // update the best length if we've improved it
    if (current_length > best_length) {
      best_length = current_length;
      best_j = j;
    }
  }
  return s.substr(best_j, best_length);
}

bool check_string(std::string s, int N) {
  std::bitset<128> bits;
  bits.reset();
  for (auto c : s) {
    bits.set(c);
  }
  return bits.count() <= N;
}

#pragma GCC diagnostic pop

///////////
// Tests //
///////////

TEST(maxnsubstr, maxnsubstr_chris) {
  ASSERT_EQ("ddddd", maxnsubstr_chris("abcddddd", 1));
  ASSERT_EQ("cddddd", maxnsubstr_chris("abcddddd", 2));
  ASSERT_EQ("cccddddd", maxnsubstr_chris("abcccddddd", 2));
  ASSERT_EQ("bcccddddd", maxnsubstr_chris("abcccddddd", 3));
  ASSERT_EQ("abcccddddd", maxnsubstr_chris("abcccddddd", 10));
  ASSERT_EQ("", maxnsubstr_chris("abcddddd", 0));
}

TEST(maxnsubstr, adam) {
  ASSERT_EQ("ddddd", maxnsubstr_adam("abcddddd", 1));
  ASSERT_EQ("cddddd", maxnsubstr_adam("abcddddd", 2));
  ASSERT_EQ("cccddddd", maxnsubstr_adam("abcccddddd", 2));
  ASSERT_EQ("bcccddddd", maxnsubstr_adam("abcccddddd", 3));
  ASSERT_EQ("abcccddddd", maxnsubstr_adam("abcccddddd", 10));
  ASSERT_EQ("", maxnsubstr_adam("abcddddd", 0));
}

TEST(maxnsubstr, christophe) {
  ASSERT_EQ("ddddd", maxnsubstr_christophe("abcddddd", 1));
  ASSERT_EQ("cddddd", maxnsubstr_christophe("abcddddd", 2));
  ASSERT_EQ("cccddddd", maxnsubstr_christophe("abcccddddd", 2));
  ASSERT_EQ("bcccddddd", maxnsubstr_christophe("abcccddddd", 3));
  ASSERT_EQ("abcccddddd", maxnsubstr_christophe("abcccddddd", 10));
  ASSERT_EQ("", maxnsubstr_christophe("abcddddd", 0));
}

////////////////
// Benchmarks //
////////////////

static const size_t BM_length_min = 8;
static const size_t BM_length_max = 10 << 10;

void baseline(benchmark::State& state) {
  std::string t(static_cast<size_t>(state.range(0)), 'a');

  while (state.KeepRunning()) {
    for (auto& c : t)  // generate string
      c = rand_r(&seed) % std::numeric_limits<char>::max();
    benchmark::DoNotOptimize(t.data());
  }
}
BENCHMARK(baseline)->Range(BM_length_min, BM_length_max);

void chris(benchmark::State& state) {
  std::string t(static_cast<size_t>(state.range(0)), 'a');

  while (state.KeepRunning()) {
    for (auto& c : t)  // generate string
      c = rand_r(&seed) % std::numeric_limits<char>::max();

    maxnsubstr_chris(
        t, (static_cast<unsigned long>(rand_r(&seed)) % t.length()) / 2);
    benchmark::DoNotOptimize(t.data());
  }
}
BENCHMARK(chris)->Range(BM_length_min, BM_length_max);

void adam(benchmark::State& state) {
  std::string t(static_cast<size_t>(state.range(0)), 'a');

  while (state.KeepRunning()) {
    for (auto& c : t)  // generate string
      c = static_cast<unsigned long>(rand_r(&seed)) %
          std::numeric_limits<char>::max();

    maxnsubstr_adam(t,
                    static_cast<int>(static_cast<unsigned long>(rand_r(&seed)) %
                                     t.length()) /
                        2);
    benchmark::DoNotOptimize(t.data());
  }
}
BENCHMARK(adam)->Range(BM_length_min, BM_length_max);

void christophe(benchmark::State& state) {
  std::string t(static_cast<size_t>(state.range(0)), 'a');

  while (state.KeepRunning()) {
    for (auto& c : t)  // generate string
      c = rand_r(&seed) % std::numeric_limits<char>::max();

    maxnsubstr_christophe(
        t, static_cast<int>(static_cast<unsigned long>(rand_r(&seed)) %
                            t.length()) /
               2);
    benchmark::DoNotOptimize(t.data());
  }
}
BENCHMARK(christophe)->Range(BM_length_min, BM_length_max);

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  const auto ret = RUN_ALL_TESTS();
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  return ret;
}
