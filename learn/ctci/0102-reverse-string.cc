/*
 * 1.2 Write a function void reverse(char *str) which reverses a
 * null-terminated string.
 */
#include "./ctci.h"

#include <limits>
#include <utility>

static unsigned int seed = 0xCEC;

//
// First solution. Get the length of the string, then iterate over the
// first half, swapping with the second half.
//
// O(n) time, O(1) space.
//
void reverse(char *str) {
  const auto len = strlen(str) - 1;
  for (size_t i = 0; i <= len / 2; i++)
    std::swap(str[i], str[len - i]);  // NOLINT(build/include_what_you_use)
}

///////////
// Tests //
///////////

TEST(Reverse, reverse) {
  char test1[] = "abcdefg";
  char test2[] = "abcdefg ";

  reverse(test1);
  reverse(test2);

  ASSERT_STREQ("gfedcba", test1);
  ASSERT_STREQ(" gfedcba", test2);
}

////////////////
// Benchmarks //
////////////////

static const size_t BM_length_min = 8;
static const size_t BM_length_max = 10 << 10;

void BM_reverse(benchmark::State &state) {
  const auto len = static_cast<size_t>(state.range(0));
  char *t = new char[len];

  for (size_t i = 0; i < len; i++)
    t[i] = rand_r(&seed) % std::numeric_limits<char>::max();

  while (state.KeepRunning()) {
    // FIXME(cec): Segmentation fault!
    // reverse(t);
    benchmark::DoNotOptimize(t[0]);
  }

  delete[] t;
}
BENCHMARK(BM_reverse)->Range(BM_length_min, BM_length_max);

CTCI_MAIN();
