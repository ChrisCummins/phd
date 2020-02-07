// Given a collection of intervals, merge all overlapping intervals.
//
// For example
// Given [1, 3], [2, 6], [8, 10], [15, 18]
// Return [1, 6], [8, 10], [15, 18]

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include "labm8/cpp/test.h"

using std::max;
using std::min;
using std::pair;
using std::vector;

// Time: O(n log n)
// Space: O(n)
vector<pair<int, int>> MergeIntervals(const vector<pair<int, int>>& in) {
  vector<pair<int, int>> intervals(in);
  std::sort(intervals.begin(), intervals.end());

  vector<pair<int, int>> merged;

  for (size_t i = 0; i < intervals.size(); ++i) {
    size_t lastMerged = merged.size() - 1;

    if (!i || intervals[i].first > merged[lastMerged].second) {
      merged.push_back(intervals[i]);
    } else {
      merged[lastMerged].second =
          max(merged[lastMerged].second, intervals[i].second);
    }
  }

  return merged;
}

TEST(MergeIntervals, EmptyInput) {
  vector<pair<int, int>> input = {};
  vector<pair<int, int>> expected = {};
  EXPECT_EQ(MergeIntervals(input), expected);
}

TEST(MergeIntervals, SingleElement) {
  vector<pair<int, int>> input = {{1, 3}};
  vector<pair<int, int>> expected = {{1, 3}};
  EXPECT_EQ(MergeIntervals(input), expected);
}

TEST(MergeIntervals, OverlappingRegion) {
  vector<pair<int, int>> input = {{1, 3}, {2, 6}};
  vector<pair<int, int>> expected = {{1, 6}};
  EXPECT_EQ(MergeIntervals(input), expected);
}

TEST(MergeIntervals, NonOverlappingRegion) {
  vector<pair<int, int>> input = {{1, 3}, {4, 6}};
  vector<pair<int, int>> expected = {{1, 3}, {4, 6}};
  EXPECT_EQ(MergeIntervals(input), expected);
}

TEST(MergeIntervals, NonOverlappingRegionRever) {
  vector<pair<int, int>> input = {{4, 6}, {1, 3}};
  vector<pair<int, int>> expected = {{1, 3}, {4, 6}};
  EXPECT_EQ(MergeIntervals(input), expected);
}

TEST(MergeIntervals, ExampleInput) {
  vector<pair<int, int>> input = {{1, 3}, {2, 6}, {8, 10}, {15, 18}};
  vector<pair<int, int>> expected = {{1, 6}, {8, 10}, {15, 18}};
  EXPECT_EQ(MergeIntervals(input), expected);
}

TEST(MergeIntervals, AdversarialInput) {
  vector<pair<int, int>> input = {{2, 3}, {4, 5}, {6, 7}, {8, 9}, {1, 10}};
  vector<pair<int, int>> expected = {{1, 10}};
  EXPECT_EQ(MergeIntervals(input), expected);
}

TEST_MAIN();
