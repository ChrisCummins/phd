// This problem was asked by Amazon.
//
// Given a sorted array arr of distinct integers, return the lowest index i for
// which arr[i] == i. Return null if there is no such index.
//
// For example, given the array [-5, -3, 2, 3], return 2 since arr[2] == 2.
// Even though arr[3] == 3, we return 2 since it's the lowest index.
#include <vector>
#include "labm8/cpp/test.h"

template <typename T>
int F(const std::vector<T>& v) {
  for (size_t i = 0; i < v.size(); ++i) {
    if (static_cast<size_t>(v[i]) == i) {
      return i;
    }
  }

  return -1;
}

TEST(FindLowestMatchingIndex, EmptyList) { EXPECT_EQ(F<int>({}), -1); }

TEST(FindLowestMatchingIndex, SingleElementListWithNoMatches) {
  EXPECT_EQ(F<int>({3}), -1);
}

TEST(FindLowestMatchingIndex, ListWithMatchAtFirstPosition) {
  EXPECT_EQ(F<int>({0, 1, 2}), 0);
}

TEST(FindLowestMatchingIndex, ListWithMatchAtSecondPosition) {
  EXPECT_EQ(F<int>({1, 1, 2}), 1);
}

TEST(FindLowestMatchingIndex, ExampleInput) {
  EXPECT_EQ(F<int>({-5, -3, 2, 3}), 2);
}

TEST_MAIN();
