/*
 * Write a method to perform basic string compression using the counts
 * of repeated characters. For example, the string aabcccccaaa would
 * become a2b1c5a3. If the "compressed" string would not become
 * smaller than the original string, your method should return the
 * original string.
 */
#include "./ctci.h"

#include <ostream>
#include <string>

static unsigned int seed = 0xCEC;

//
// A stateful solution which iterates over the string, storing the
// current character and the current character count. If the new
// character does not match the current character, output the
// character and its count.
//
// O(n) time, O(n) space.
//
std::string compress_str(const std::string &str) {
  std::ostringstream oss;
  char curr_c = '\0';
  size_t count = 0;

  for (auto &c : str) {
    if (c == curr_c) {
      count++;
    } else {
      if (curr_c != '\0') oss << curr_c << count;
      curr_c = c;
      count = 1;
    }
  }

  // Output last character:
  if (curr_c != '\0') oss << curr_c << count;

  // Return the shortest string:
  std::string os = oss.str();
  return os.size() < str.size() ? os : str;
}

///////////
// Tests //
///////////

TEST(Permutation, compress_str) {
  const auto in1 = std::string("aabcccccaaa");
  auto out1 = compress_str(in1);
  ASSERT_EQ(std::string("a2b1c5a3"), out1);

  const auto in2 = std::string("abcde");
  auto out2 = compress_str(in2);
  ASSERT_EQ(std::string("abcde"), out2);
}

////////////////
// Benchmarks //
////////////////

static const size_t BM_length_min = 8;
static const size_t BM_length_max = 10 << 10;

void BM_compress_str(benchmark::State &state) {
  const auto strlen = static_cast<size_t>(state.range(0));
  std::string t;

  while (state.KeepRunning()) {
    for (size_t i = 0; i < strlen; i++)
      t += static_cast<char>(rand_r(&seed) % 5);

    std::string o = compress_str(t);
    benchmark::DoNotOptimize(o[0]);
  }
}
BENCHMARK(BM_compress_str)->Range(BM_length_min, BM_length_max);

CTCI_MAIN();
