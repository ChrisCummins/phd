// This problem was asked by Facebook.
//
// Describe an algorithm to compute the longest increasing subsequence of an
// array of numbers in O(n log n) time.

#include <vector>

#include "labm8/cpp/test.h"

// Time: O(n)
// Space: O(1)
size_t f(const std::vector<int>& V) {
  if (V.size() < 2) {
    return V.size();
  }
  size_t a = 0;
  size_t mn = 0;
  for (size_t b = 1; b < V.size(); ++b) {
    if (V[b - 1] > V[b]) {
      size_t n = b - a;
      if (n > mn) {
        mn = n;
      }
      a = b;
    }
  }

  size_t n = V.size() - a;
  if (n > mn) {
    mn = n;
  }

  return mn;
}

TEST(LongestAscendingSubsequence, EmptyList) { EXPECT_EQ(f({}), 0); }

TEST(LongestAscendingSubsequence, SingleElementList) { EXPECT_EQ(f({1}), 1); }

TEST(LongestAscendingSubsequence, TwoElementList) { EXPECT_EQ(f({1, 2}), 2); }

TEST(LongestAscendingSubsequence, TwoElementDescendingList) {
  EXPECT_EQ(f({1, -1}), 1);
}

TEST(LongestAscendingSubsequence, ThreeElementList) {
  EXPECT_EQ(f({1, 2, 3}), 3);
}

TEST(LongestAscendingSubsequence, FourElementList) {
  EXPECT_EQ(f({0, 1, 2, -1}), 3);
}

TEST(LongestAscendingSubsequence, SixElementList) {
  EXPECT_EQ(f({1, 2, 0, 1, 2, 3}), 4);
}

TEST_MAIN();
