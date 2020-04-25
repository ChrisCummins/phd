// Given an array nums containing n + 1 integers where each integer is between 1
// and n (inclusive), prove that at least one duplicate number must exist.
// Assume that there is only one duplicate number and find the duplicate one.

#include <vector>

#include "labm8/cpp/test.h"

using std::vector;

// Time: O(n)
// Space: O(n)
int FindDupe(const vector<int>& x) {
  if (!x.size()) {
    return -1;
  }

  vector<bool> visited(x.size(), false);

  for (auto val : x) {
    // Sanity check.
    if (val >= visited.size() || val < 1) {
      return -1;
    }

    // We have already visited this guy.
    if (visited[val]) {
      return val;
    }
    // Mark the guy as visited.
    visited[val] = 1;
  }

  return -1;
}

TEST(FindDupe, EmptyList) { EXPECT_EQ(FindDupe({}), -1); }

TEST(FindDupe, NoDupe) {
  EXPECT_EQ(FindDupe({1}), -1);
  EXPECT_EQ(FindDupe({1, 2, 3}), -1);
}

TEST(FindDupe, Examples) {
  EXPECT_EQ(FindDupe({1, 2, 2, 3}), 2);
  EXPECT_EQ(FindDupe({1, 2, 3, 4, 5, 6, 7, 8, 3}), 3);
}

TEST_MAIN();
