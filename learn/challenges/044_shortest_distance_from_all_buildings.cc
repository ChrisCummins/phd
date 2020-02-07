// You want to build a house on an empty land which reaches all buildings in the
// shortest amount of distance. You can only move up, down, left and right. You
// are given a 2D grid of values 0, 1, 2, where:
//
//   1. Each 0 marks an empty land which you can pass by freely.
//   2. Each 1 marks a building which you cannot pass through.
//   3. Each 2 marks an obstacle which you cannot pass through.
//
// For example, given three buildings at (0,0), (0,4), (2,2), and an obstacle at
// (0,2):
//
//     1 - 0 - 2 - 0 - 1
//     |   |   |   |   |
//     0 - 0 - 0 - 0 - 0
//     |   |   |   |   |
//     0 - 0 - 1 - 0 - 0
//
// The point (1,2) is an ideal empty land to build a house, as the total travel
// distance of 3+3+1=7 is minimal. So return 7.
//
// Note: There will be at least one building. If is not possible to build such
// house according to the above rules, return -1.
#include <deque>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "labm8/cpp/test.h"

using std::deque;
using std::numeric_limits;
using std::pair;
using std::unordered_set;
using std::vector;

using Board = vector<vector<int>>;
using Point = pair<int, int>;

// Find an empty square.
Point StartingPoint(const Board& b) {
  const int n = b.size();
  Point p{0, 0};

  while (p.first < n && p.second < n) {
    if (b[p.first][p.second] != 0) {
      ++p.first;
      if (p.first >= n) {
        ++p.second;
        p.first = 0;
      }
    }
  }

  // No valid starting point.
  return {-1, -1};
}

// Return vector of neighboring empty squares.
//
// Time: O(1)
// Space: O(1)
vector<Point> Neighbors(const Point& p, const Board& b) {
  const int n = b.size();
  const int m = b[0].size();
  vector<Point> ne;

  Point x;

  x = {p.first - 1, p.second};
  if (x.first >= 0 && b[x.first][x.second] == 0) {
    ne.push_back(x);
  }

  x = {p.first + 1, p.second};
  if (x.first < n && b[x.first][x.second] == 0) {
    ne.push_back(x);
  }

  x = {p.first, p.second - 1};
  if (x.second >= 0 && b[x.first][x.second] == 0) {
    ne.push_back(x);
  }

  x = {p.first, p.second + 1};
  if (x.second < m && b[x.first][x.second] == 0) {
    ne.push_back(x);
  }

  return ne;
}

// Time: O(n)
// Space: O(n)
int Distance(const Board& b, const Point& start, const Point& end) {
  unordered_set<int> v;
  deque<pair<Point, int>> q;
  q.push_back({start, 0});

  while (!q.empty()) {
    auto tmp = q.front();
    Point p = tmp.first;
    int d = tmp.second;
    q.pop_front();

    if (p == end) {
      return d;
    }

    v.insert(p.first * b[0].size() + p.second);

    for (auto ne : Neighbors(p, b)) {
      if (v.find(ne.first * b[0].size() + ne.second) == v.end()) {
        q.push_back({ne, d + 1});
      }
    }
  }

  return -1;
}

// Time: O(n ^ 2)
// Space: O(n)
int ComputeScore(const Board& b, const Point& p) {
  int score = 0;

  for (int i = 0; i < b.size(); ++i) {
    for (int j = 0; j < b[0].size(); ++j) {
      if (b[i][j] == 1) {
        int d = Distance(b, {i, j}, p);
        if (d == -1) {
          return -1;
        }
        score += d;
      }
    }
  }

  return score;
}

// Time: O(n ^ 3)
// Space: O(n)
Point IdealSpot(const Board& b) {
  // Check dimensionality of board.
  const int n = b.size();
  if (!n) {
    return {-1, -1};
  }
  const int m = b[0].size();
  for (const auto& row : b) {
    if (row.size() != m) {
      return {-1, -1};
    }
  }

  Point bestPoint{-1, -1};
  int bestScore = numeric_limits<int>::max();

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      if (b[i][j] == 0) {
        int score = ComputeScore(b, {i, j});
        if (score != -1 && score <= bestScore) {
          bestPoint = {i, j};
          bestScore = score;
        }
      }
    }
  }

  return bestPoint;
}

TEST(ShortestDistanceFromAllBuildings, EmptyBoard) {
  Board b{};
  Point expected = {-1, -1};
  Point actual = IdealSpot(b);

  EXPECT_EQ(actual, expected);
}

TEST(ShortestDistanceFromAllBuildings, SingleRowBoard) {
  Board b{{1, 0, 1}};
  Point expected = {0, 1};
  Point actual = IdealSpot(b);

  EXPECT_EQ(actual, expected);
}

TEST(ShortestDistanceFromAllBuildings, SingleColumnBoard) {
  Board b{{1}, {0}, {1}};
  Point expected = {1, 0};
  Point actual = IdealSpot(b);

  EXPECT_EQ(actual, expected);
}

TEST(ShortestDistanceFromAllBuildings, ExampleInput) {
  Board b{
      {1, 0, 2, 0, 1},
      {0, 0, 0, 0, 0},
      {0, 0, 1, 0, 0},
  };
  Point expected = {1, 2};
  Point actual = IdealSpot(b);

  EXPECT_EQ(actual, expected);
}

TEST_MAIN();
