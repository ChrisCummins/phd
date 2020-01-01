// By starting at the top of the triangle below and moving to adjacent numbers
// on the row below, the maximum total from top to bottom is 23.
//
// 3
// 7 4
// 2 4 6
// 8 5 9 3
//
// That is, 3 + 7 + 4 + 9 = 23.
//
// Find the maximum total from top to bottom of the triangle below:
//
// 75
// 95 64
// 17 47 82
// 18 35 87 10
// 20 04 82 47 65
// 19 01 23 75 03 34
// 88 02 77 73 07 63 67
// 99 65 04 28 06 16 70 92
// 41 41 26 56 83 40 80 70 33
// 41 48 72 33 47 32 37 16 94 29
// 53 71 44 65 25 43 91 52 97 51 14
// 70 11 33 28 77 73 17 78 39 68 17 57
// 91 71 52 38 17 14 91 43 58 50 27 29 48
// 63 66 04 68 89 53 67 30 73 16 69 87 40 31
// 04 62 98 27 23 09 70 98 73 93 38 53 60 04 23
//
// NOTE: As there are only 16384 routes, it is possible to solve this problem by
// trying every route. However, Problem 67, is the same challenge with a
// triangle containing one-hundred rows; it cannot be solved by brute force,
// and requires a clever method! ;o)
#include <algorithm>
#include <deque>
#include <iostream>
#include <vector>

#include "labm8/cpp/logging.h"

// A helper struct for storing a point in a vector-of-vectors triangle path.
// The j and i values store the vector indices for the vertical and horizontal
// position, respectively. The sum is the total sum of the values along this
// path, excluding the current point.
struct PathPoint {
  size_t j;
  size_t i;
  long long sum;
};

// Find the maximum sum path by enumerating all routes through breadth first
// traversal of the triangle. For a triangle with n levels, this take
// O(2 ^ n) time and O(n) space.
long long GetMaximumSumPath(const std::vector<std::vector<int>>& triangle) {
  long long bestSum = 0;

  // An empty triangle has no max.
  if (!triangle.size()) {
    return bestSum;
  }

  std::deque<PathPoint> q{{0, 0, 0}};
  while (q.size()) {
    const PathPoint& point = q.front();

    // Calculate the sum at this point in the path.
    int val = triangle[point.j][point.i];
    long long sum = point.sum + static_cast<long long>(val);

    if (point.j < triangle.size() - 1) {
      CHECK(triangle[point.j + 1].size() == triangle[point.j].size() + 1);

      q.push_back({point.j + 1, point.i, sum});
      q.push_back({point.j + 1, point.i + 1, sum});
    } else {
      bestSum = std::max(sum, bestSum);
    }

    q.pop_front();
  }

  return bestSum;
}

int main() {
  std::vector<std::vector<int>> triangle;

  triangle.push_back({75});
  triangle.push_back({95, 64});
  triangle.push_back({17, 47, 82});
  triangle.push_back({18, 35, 87, 10});
  triangle.push_back({20, 4, 82, 47, 65});
  triangle.push_back({19, 1, 23, 75, 3, 34});
  triangle.push_back({88, 2, 77, 73, 7, 63, 67});
  triangle.push_back({99, 65, 4, 28, 6, 16, 70, 92});
  triangle.push_back({41, 41, 26, 56, 83, 40, 80, 70, 33});
  triangle.push_back({41, 48, 72, 33, 47, 32, 37, 16, 94, 29});
  triangle.push_back({53, 71, 44, 65, 25, 43, 91, 52, 97, 51, 14});
  triangle.push_back({70, 11, 33, 28, 77, 73, 17, 78, 39, 68, 17, 57});
  triangle.push_back({91, 71, 52, 38, 17, 14, 91, 43, 58, 50, 27, 29, 48});
  triangle.push_back({63, 66, 4, 68, 89, 53, 67, 30, 73, 16, 69, 87, 40, 31});
  triangle.push_back({4, 62, 98, 27, 23, 9, 70, 98, 73, 93, 38, 53, 60, 4, 23});

  auto max = GetMaximumSumPath(triangle);
  std::cout << max << std::endl;

  assert(max == 1074);
}
