// This problem was asked by Microsoft.
//
// Given a string, generate all possible subsequences of the string.
//
// For example, given the string xyz, return an array or set with the following
// strings:
//
//     x
//     y
//     z
//     xy
//     xz
//     yz
//     xyz
//
// Note that zx is not a valid subsequence since it is not in the order of the
// given string.

#include <unordered_set>
#include "labm8/cpp/string.h"
#include "labm8/cpp/test.h"

using std::unordered_set;

// Time: O(2 ^ n * n)
// Space: O(2 ^ n * n)  - could be O(n) using a generator pattern
unordered_set<string> GenerateSubsequences(const string& S) {
  unordered_set<string> ss;
  int n = (1 << S.size()) - 1;
  ss.reserve(n);

  for (int i = 1; i <= n; ++i) {
    string x;
    int ii = i;
    for (size_t j = 0; j < S.size(); ++j) {
      if (ii & 1) {
        x.push_back(S[j]);
      }
      ii >>= 1;
    }

    ss.insert(x);
  }

  return ss;
}

TEST(GenerateSubsequences, EmptyString) {
  auto ss = GenerateSubsequences("");
  EXPECT_EQ(ss.size(), 0);
}

TEST(GenerateSubsequences, SingleChar) {
  auto ss = GenerateSubsequences("a");
  EXPECT_EQ(ss.size(), 1);
  EXPECT_TRUE(ss.find("a") != ss.end());
}

TEST(GenerateSubsequences, DoubleChar) {
  auto ss = GenerateSubsequences("aa");
  EXPECT_EQ(ss.size(), 2);
  EXPECT_TRUE(ss.find("a") != ss.end());
  EXPECT_TRUE(ss.find("aa") != ss.end());
}

TEST(GenerateSubsequences, TwoDifferentChars) {
  auto ss = GenerateSubsequences("ab");
  EXPECT_EQ(ss.size(), 3);
  EXPECT_TRUE(ss.find("a") != ss.end());
  EXPECT_TRUE(ss.find("b") != ss.end());
  EXPECT_TRUE(ss.find("ab") != ss.end());
}

TEST(GenerateSubsequences, ExampleInput) {
  auto ss = GenerateSubsequences("abc");
  EXPECT_EQ(ss.size(), 7);
  EXPECT_TRUE(ss.find("a") != ss.end());
  EXPECT_TRUE(ss.find("b") != ss.end());
  EXPECT_TRUE(ss.find("c") != ss.end());
  EXPECT_TRUE(ss.find("ab") != ss.end());
  EXPECT_TRUE(ss.find("bc") != ss.end());
  EXPECT_TRUE(ss.find("ac") != ss.end());
  EXPECT_TRUE(ss.find("abc") != ss.end());
}

TEST_MAIN();
