#include "labm8/cpp/test.h"

#include <vector>

// Given a list of numbers and a number k, return whether any two numbers from
// the list add up to k. For example, given [10, 15, 3, 7] and k of 17, return
// true since 10 + 7 is 17. Bonus: Can you do this in one pass?
bool AddUpToK(const std::vector<int>& vec, const int k) {
  std::set<int> visited;

  for (int i : vec) {
    int complement = k - i;
    if (visited.find(complement) == visited.end()) {
      visited.insert(i);
    } else {
      return true;
    }
  }

  return false;
}

TEST(AddUpToK, EmptyListReturnsFalse) {
  std::vector<int> v;
  EXPECT_FALSE(AddUpToK(v, 0));
}

TEST(AddUpToK, SingleElementListReturnsFalse) {
  std::vector<int> v{1};
  EXPECT_FALSE(AddUpToK(v, 0));
}

TEST(AddUpToK, PairOfElementsWithMatchingK) {
  std::vector<int> v{1, 2};
  EXPECT_TRUE(AddUpToK(v, 3));
}

TEST(AddUpToK, PairOfElementsWithUnmatchingK) {
  std::vector<int> v{1, 2};
  EXPECT_FALSE(AddUpToK(v, 5));
}

TEST_MAIN();
