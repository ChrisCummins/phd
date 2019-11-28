/*
 * 1.1 Implement an algorithm to determine if a string has all unique
 * characters. What if you cannot use additional data structures?
 */
#include "./ctci.h"

#include <bitset>
#include <limits>
#include <string>
#include <unordered_set>

static unsigned int seed = 0xCEC;

//
// First solution. Generic solution which works for all container
// types. Use a set to store whether an element has already appeared
// in the container.
//
// O(n) time, O(n) space.
//
template <typename Container>
bool unique(const Container &cont) {
  std::unordered_set<typename Container::value_type> set;

  for (auto &elem : cont)
    if (set.find(elem) != set.end())
      return false;
    else
      set.insert(elem);

  return true;
}

//
// Second solution. By specializing to strings, we can remove the
// expensive heap allocated set and replace it with a bitset, one bit
// for every possible character value. To check whether a string is
// unique we then iterate over the characters and index into this
// bitset.
//
// O(n) time, O(1) space.
//
bool str_unique(const std::string &str) {
  std::bitset<std::numeric_limits<char>::max()> chars = {0};

  for (auto &c : str)
    if (chars[static_cast<size_t>(c)])
      return false;
    else
      chars[static_cast<size_t>(c)] = true;

  return true;
}

//
// Additional constraint: no extra data structures. This requires us
// to iterate over every the remaining element of the container,
// comparing equality with each. E.g.:
//
//   [0]  1  2  3  4  5  ...
//   [1]     2  3  4  5  ...
//   [2]        3  4  5  ...
//   [3]           4  5  ...
//
// O(n^2) time, O(1) space.
//
template <typename Container>
bool inplace_unique(const Container &cont) {
  auto left = cont.begin();

  while (left != cont.end()) {
    auto right = left + 1;
    while (right != cont.end())
      if (*left == *right++) return false;
    left++;
  }

  return true;
}

////////////////
// Unit tests //
////////////////

static const std::string TEST_is_unique("abcdefg ");
static const std::string TEST_not_unique("abcdefga");

TEST(Unique, unique) {
  ASSERT_TRUE(unique(TEST_is_unique));
  ASSERT_FALSE(unique(TEST_not_unique));
}

TEST(Unique, str_unique) {
  ASSERT_TRUE(str_unique(TEST_is_unique));
  ASSERT_FALSE(str_unique(TEST_not_unique));
}

TEST(Unique, inplace_unique) {
  ASSERT_TRUE(inplace_unique(TEST_is_unique));
  ASSERT_FALSE(inplace_unique(TEST_not_unique));
}

////////////////
// Benchmarks //
////////////////

static const size_t BM_length_min = 8;
static const size_t BM_length_max = 10 << 10;

void BM_unique(benchmark::State &state) {
  std::string t(static_cast<size_t>(state.range(0)), 'a');

  while (state.KeepRunning()) {
    for (auto &c : t) c = rand_r(&seed) % std::numeric_limits<char>::max();

    unique(t);
    benchmark::DoNotOptimize(t.data());
  }
}
BENCHMARK(BM_unique)->Range(BM_length_min, BM_length_max);

void BM_str_unique(benchmark::State &state) {
  std::string t(static_cast<size_t>(state.range(0)), 'a');

  while (state.KeepRunning()) {
    for (auto &c : t) c = rand_r(&seed) % std::numeric_limits<char>::max();

    str_unique(t);
    benchmark::DoNotOptimize(t.data());
  }
}
BENCHMARK(BM_str_unique)->Range(BM_length_min, BM_length_max);

void BM_inplace_unique(benchmark::State &state) {
  std::string t(static_cast<size_t>(state.range(0)), 'a');

  while (state.KeepRunning()) {
    for (auto &c : t) c = rand_r(&seed) % std::numeric_limits<char>::max();

    inplace_unique(t);
    benchmark::DoNotOptimize(t.data());
  }
}
BENCHMARK(BM_inplace_unique)->Range(BM_length_min, BM_length_max);

CTCI_MAIN();
