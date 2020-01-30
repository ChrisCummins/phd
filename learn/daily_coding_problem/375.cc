// This problem was asked by Google.
//
// The h-index is a metric used to measure the impact and productivity
// of a scientist or researcher.
//
// A scientist has index h if h of their N papers have at least h
// citations each, and the other N - h papers have no more than h
// citations each. If there are multiple possible values for h, the
// maximum value is used.
//
// Given an array of natural numbers, with each value representing the
// number of citations of a researcher's paper, return the h-index of
// that researcher.
//
// For example, if the array was:
//
//     [4, 0, 0, 2, 3]
//
// This means the researcher has 5 papers with 4, 1, 0, 2, and 3
// citations respectively. The h-index for this researcher is 2, since
// they have 2 papers with at least 2 citations and the remaining 3
// papers have no more than 2 citations.
#include "labm8/cpp/test.h"

#include <algorithm>
#include <vector>

using std::max_element;
using std::vector;

int HIndex(const vector<int>& cit) {
  auto max = max_element(cit.begin(), cit.end());
  if (max == cit.end()) {
    return 0;
  }
  vector<int> f(*max + 1, 0);

  for (const auto& p : cit) {
    f[p] += 1;
  }

  for (int i = f.size() - 1; i >= 0; --i) {
    if (i < f.size() - 1) {
      f[i] += f[i + 1];
    }
    if (f[i] >= i) {
      return i;
    }
  }

  return 0;
}

TEST(HIndex, EmptyList) { EXPECT_EQ(HIndex({}), 0); }

TEST(HIndex, OneList) { EXPECT_EQ(HIndex({1}), 1); }

TEST(HIndex, ZeroList) { EXPECT_EQ(HIndex({0}), 0); }

TEST(HIndex, DoubleZeroList) { EXPECT_EQ(HIndex({0, 0}), 0); }

TEST(HIndex, DoubleOneList) { EXPECT_EQ(HIndex({1, 1}), 1); }

TEST(HIndex, DoubleTwoList) { EXPECT_EQ(HIndex({2, 2}), 2); }

TEST(HIndex, ThreeList) { EXPECT_EQ(HIndex({10, 100, 3}), 3); }

TEST(HIndex, ThreeListTwo) { EXPECT_EQ(HIndex({10, 100, 0}), 2); }

TEST(HIndex, ExampleInput) { EXPECT_EQ(HIndex({4, 0, 0, 2, 3}), 2); }

TEST_MAIN();
