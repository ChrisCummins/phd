// This problem was asked by Google.
//
// PageRank is an algorithm used by Google to rank the importance of
// different websites. While there have been changes over the years,
// the central idea is to assign each site a score based on the
// importance of other pages that link to that page.
//
// More mathematically, suppose there are N sites, and each site i has
// a certain count Ci of outgoing links. Then the score for a
// particular site Sj is defined as :
//
// score(Sj) = (1 - d) / N + d * (score(Sx) / Cx+ score(Sy) / Cy+ ... +
// score(Sz) / Cz))
//
// Here, Sx, Sy, ..., Sz denote the scores of all the other sites that
// have outgoing links to Sj, and d is a damping factor, usually set
// to around 0.85, used to model the probability that a user will stop
// searching.
//
// Given a directed graph of links between various websites, write a
// program that calculates each site's page rank.
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>
#include "labm8/cpp/test.h"

using std::max;
using std::vector;

// Args:
//   C: A matrix where a row C[j] represents site j, a column C[j][i] is the
//     number of outgoing links from site i, iff site i links to site j. If site
//     i does not link to site j, C[j][i] == 0.
//
//   d: The damping factor.
//
//   T: The maximum number of iterations to perform. Possible extension would
//     be to terminate once the delta from one timestep to the next falls below
//     a threshold, or if the relative order of sites does not change from one
//     iteration to the next (though this would require an expensive O(n log n)
//     sort.
//
// Time: O(T * N^2) = O(n ^ 2)
// Space: O(T * N), but could easily be made O(2 * N) by double buffering rather
// than appending to a matrix.
vector<float> F(const vector<vector<int>>& C, float d, int T) {
  int N = C.size();
  float t0 = (1 - d) / static_cast<float>(N);
  vector<vector<float>> R({vector<float>(N)});

  std::cout << std::fixed;
  std::cout << std::setprecision(3);

  // init t0
  for (int i = 0; i < N; ++i) {
    R[0][i] = t0;
  }

  for (int t = 1; t < T; ++t) {
    R.push_back(vector<float>(N));
    std::cout << "R[t" << t << "] = ";
    for (int i = 0; i < N; ++i) {
      R[t][i] = t0;
      for (int j = 0; j < N; ++j) {
        if (C[i][j]) {
          R[t][i] += R[t - 1][j] / C[i][j];
        }
      }
      std::cout << R[t][i] << " ";
    }
    std::cout << std::endl;
  }

  return R[max(T - 1, 0)];
}

TEST(PageRank, ThreeByThree) {
  vector<vector<int>> C({
      {0, 1, 1},
      {2, 0, 0},
      {2, 1, 0},
  });

  auto r = F(C, 0.85, 5);

  EXPECT_NEAR(r[0], 0.525, 0.0005);
  EXPECT_NEAR(r[1], 0.237, 0.0005);
  EXPECT_NEAR(r[2], 0.412, 0.0005);
}

TEST_MAIN();
