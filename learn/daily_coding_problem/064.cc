// This problem was asked by Google.
//
// A knight's tour is a sequence of moves by a knight on a chessboard such that
// all squares are visited once.
//
// Given N, write a function to return the number of knight's tours on an N by
// N chessboard.
#include <stack>
#include <unordered_set>
#include <utility>
#include <vector>
#include "labm8/cpp/port.h"
#include "labm8/cpp/test.h"

using labm8::uint64;
using std::find;
using std::pair;
using std::stack;
using std::unordered_set;
using std::vector;

static const vector<pair<int, int>> moves({
    {-2, 1},
    {-1, 2},
    {1, 2},
    {2, 1},
    {2, -1},
    {1, -2},
    {-1, -2},
    {-2, -1},
});

// Time: O(8 ^ n) !!! i think
// Space: O(n ^ 2)
uint64 KnightsTours(const int n) {
  uint64 c = 0;
  stack<pair<int, unordered_set<int>>> q;

  for (int i = 0; i < n * n; ++i) {
    q.push({i, {i}});
  }

  while (!q.empty()) {
    pair<int, unordered_set<int>> x(q.top());
    q.pop();
    if (x.second.size() >= n * n) {
      ++c;
    } else {
      for (auto& move : moves) {
        int j = (x.first / n) + move.first;
        int i = (x.first % n) + move.second;
        if (j >= 0 && j < n && i >= 0 && i < n) {
          auto z = j * n + i;
          if (x.second.find(z) == x.second.end()) {
            auto v = x.second;
            v.insert(z);
            q.push({z, v});
          }
        }
      }
    }
  }

  return c;
}

TEST(KnightsTour, Empty) { EXPECT_EQ(0, KnightsTours(0)); }

TEST(KnightsTour, One) { EXPECT_EQ(1, KnightsTours(1)); }

TEST(KnightsTour, Two) { EXPECT_EQ(0, KnightsTours(2)); }

TEST(KnightsTour, Three) { EXPECT_EQ(0, KnightsTours(3)); }

TEST(KnightsTour, Four) { EXPECT_EQ(0, KnightsTours(4)); }

// TEST(KnightsTour, Five) { EXPECT_EQ(1728, KnightsTours(5)); }

TEST_MAIN();
