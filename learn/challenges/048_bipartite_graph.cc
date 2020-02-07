#include <deque>
#include <unordered_set>
#include <utility>
#include <vector>

#include "labm8/cpp/logging.h"
#include "labm8/cpp/test.h"

using std::deque;
using std::pair;
using std::unordered_set;
using std::vector;

// Assumes graph is connected.
bool IsBipartite(const vector<vector<int>>& adjacencies) {
  if (!adjacencies.size()) {
    return false;
  }

  // false - black, true - red
  vector<bool> colors(adjacencies.size(), false);

  unordered_set<int> visited;

  deque<pair<int, bool>> q;
  q.push_back({0, false});

  while (!q.empty()) {
    int curr = q.front().first;
    bool color = q.front().second;
    q.pop_front();
    visited.insert(curr);
    colors[curr] = color;

    for (const auto& n : adjacencies[curr]) {
      CHECK(n >= 0 && n < adjacencies.size());

      if (visited.find(n) == visited.end()) {
        q.push_back({n, !color});
      } else if (colors[n] == color) {
        return false;
      }
    }
  }

  return true;
}

TEST(BipartiteGraph, EmptyGraph) {
  vector<vector<int>> input;

  ASSERT_EQ(IsBipartite(input), false);
}

TEST(BipartiteGraph, UnidirectionalPair) {
  vector<vector<int>> input{{1}, {}};

  ASSERT_EQ(IsBipartite(input), true);
}

TEST(BipartiteGraph, BidirectionalPair) {
  vector<vector<int>> input{{1}, {0}};

  ASSERT_EQ(IsBipartite(input), true);
}

TEST(BipartiteGraph, Triangle) {
  vector<vector<int>> input{{1, 2}, {2}, {}};

  ASSERT_EQ(IsBipartite(input), true);
}

TEST(BipartiteGraph, SelfEdge) {
  vector<vector<int>> input{
      {0},
  };

  ASSERT_EQ(IsBipartite(input), false);
}

TEST(BipartiteGraph, PairSelfEdge) {
  vector<vector<int>> input{
      {1},
      {1},
  };

  ASSERT_EQ(IsBipartite(input), false);
}

TEST(BipartiteGraph, Crossover) {
  vector<vector<int>> input{{1, 2}, {3}, {3, 1}, {}};

  ASSERT_EQ(IsBipartite(input), false);
}

TEST_MAIN();
