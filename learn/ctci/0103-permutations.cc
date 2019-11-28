/*
 * Given two strings, write a method to decide if one is a permutation
 * of another.
 */
#include "./ctci.h"

#include <array>
#include <limits>
#include <string>

static unsigned int seed = 0xCEC;

//
// First solution. Compare frequency counts for characters in the
// string.
//
// O(n) time, O(1) space. Best case O(1) time and space.
//
bool is_permutation(const std::string &a, const std::string &b) {
  if (a.size() != b.size())
    return false;  // Permutations must be the same length.

  // Create char-frequency tables:
  std::array<size_t, std::numeric_limits<char>::max()> a_freq = {{0}};
  std::array<size_t, std::numeric_limits<char>::max()> b_freq = {{0}};

  for (auto &c : a) a_freq[static_cast<size_t>(c)] += 1;
  for (auto &c : b) b_freq[static_cast<size_t>(c)] += 1;

  // Strings are permutations if char-frequency tables match:
  return a_freq == b_freq;
}

///////////
// Tests //
///////////

TEST(Permutations, is_permutation) {
  const std::string a1("abcde"), a2("efgh");
  const std::string b1("abcde"), b2("edcba");

  ASSERT_FALSE(is_permutation(a1, a2));
  ASSERT_TRUE(is_permutation(b1, b2));
}

////////////////
// Benchmarks //
////////////////

static const size_t BM_length_min = 8;
static const size_t BM_length_max = 10 << 10;

void BM_is_permutation(benchmark::State &state) {
  auto len = static_cast<size_t>(state.range(0));
  std::string t1(len, 'a'), t2(len, 'a');

  while (state.KeepRunning()) {
    for (auto &c : t1) c = rand_r(&seed) % std::numeric_limits<char>::max();
    for (auto &c : t2) c = rand_r(&seed) % std::numeric_limits<char>::max();

    auto ret = is_permutation(t1, t2);
    benchmark::DoNotOptimize(ret);
  }
}
BENCHMARK(BM_is_permutation)->Range(BM_length_min, BM_length_max);

CTCI_MAIN();
