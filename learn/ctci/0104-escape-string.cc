/*
 * Write a method to replace all spaces in a string with '%20'.
 */
#include "./ctci.h"

static unsigned int seed = 0xCEC;

//
// First solution. Iterate backwards over the string, inserting escape
// characters as required.
//
// O(n^2) time, O(1) space.
//
void escape_space(char *s, size_t len) {
  for (int i = static_cast<int>(len) - 1; i >= 0; i--) {
    if (s[i] == ' ') {
      // Make room for escape characters:
      for (int j = static_cast<int>(len) - 1; j > i; j--) s[j + 2] = s[j];

      s[i] = '%';
      s[i + 1] = '2';
      s[i + 2] = '0';
      len += 2;  // Increase length by escaped size
    }
  }
}

///////////
// Tests //
///////////

TEST(Escape, escape_space) {
  char a[] = "abcde";
  escape_space(a, 5);
  ASSERT_STREQ("abcde", a);

  char b[] = "abc de  ";
  escape_space(b, 6);
  ASSERT_STREQ("abc%20de", b);

  char c[] = "a bc de    ";
  escape_space(c, 7);
  ASSERT_STREQ("a%20bc%20de", c);
}

////////////////
// Benchmarks //
////////////////

static const size_t BM_length_min = 8;
static const size_t BM_length_max = 10 << 10;

void populateString(char *const t, const size_t strlen, size_t *len) {
  for (size_t i = 0; i < strlen; i++) {
    if (!rand_r(&seed) % 4)
      t[i] = ' ';
    else
      t[i] = static_cast<char>(rand_r(&seed));
  }

  *len = strlen;
  for (size_t i = 0; i < strlen; i++)
    if (t[i] == ' ') *len -= 2;
}

void BM_escape_space(benchmark::State &state) {
  const auto strlen = static_cast<size_t>(state.range(0));
  char *const t = new char[strlen];
  size_t len = 0;

  while (state.KeepRunning()) {
    populateString(t, strlen, &len);
    escape_space(t, len);
    benchmark::DoNotOptimize(*t);
  }

  delete[] t;
}
BENCHMARK(BM_escape_space)->Range(BM_length_min, BM_length_max);

CTCI_MAIN();
