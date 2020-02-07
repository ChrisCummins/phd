// This problem was asked by Google.
//
// Given a string, return the length of the longest palindromic subsequence in
// the string.
//
// For example, given the following string:
//
//     MAPTPTMTPA
//
// Return 7, since the longest palindromic subsequence in the string is APTMTPA.
// Recall that a subsequence of a string does not have to be contiguous!
//
// Your algorithm should run in O(n^2) time and space.
#include <algorithm>
#include <vector>

#include "labm8/cpp/string.h"
#include "labm8/cpp/test.h"

using std::max;
using std::vector;

int LongestPalindrome(const string& s) {
  if (!s.size()) {
    return 0;
  }

  vector<vector<int>> dp(s.size());
  for (size_t i = 0; i < s.size(); ++i) {
    dp[i].assign(s.size(), 0);
    dp[i][i] = 1;
  }

  for (int cl = 2; cl <= s.size(); ++cl) {
    for (int i = 0; i < s.size() - cl + 1; ++i) {
      int j = i + cl - 1;
      if (s[i] == s[j] && cl == 2) {
        dp[i][j] = 2;
      } else if (s[i] == s[j]) {
        dp[i][j] = dp[i + 1][j - 1] + 2;
      } else {
        dp[i][j] = max(dp[i][j - 1], dp[i + 1][j]);
      }
    }
  }

  return dp[0][s.size() - 1];
}

TEST(LongestPalindrome, EmptyString) { ASSERT_EQ(LongestPalindrome(""), 0); }

TEST(LongestPalindrome, ExampleString) {
  ASSERT_EQ(LongestPalindrome("MAPTPTMTPA"), 7);
}

TEST_MAIN();
