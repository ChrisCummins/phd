// This problem was asked by Amazon.
//
// There exists a staircase with N steps, and you can climb up either 1 or 2
// steps at a time. Given N, write a function that returns the number of unique
// ways you can climb the staircase. The order of the steps matters.
//
// For example, if N is 4, then there are 5 unique ways:
//
// 1, 1, 1, 1
// 2, 1, 1
// 1, 2, 1
// 1, 1, 2
// 2, 2

#include "labm8/cpp/test.h"

#include <iostream>
#include <unordered_set>
#include <vector>

// Time: O(n)
// Space: O(1)
int fn(const int n) {
  int a = 0, b = 1, c = 0;
  for (int i = 1; i <= n; ++i) {
    c = a + b;
    a = b;
    b = c;
  }

  return c;
}

TEST(StaircaseCounts, Zero) { EXPECT_EQ(fn(0), 0); }

TEST(StaircaseCounts, One) { EXPECT_EQ(fn(1), 1); }

TEST(StaircaseCounts, Two) { EXPECT_EQ(fn(2), 2); }

TEST(StaircaseCounts, Three) { EXPECT_EQ(fn(3), 3); }

TEST(StaircaseCounts, Four) { EXPECT_EQ(fn(4), 5); }

TEST(StaircaseCounts, Five) { EXPECT_EQ(fn(5), 8); }

// What if, instead of being able to climb 1 or 2 steps at a time, you could
// climb any number from a set of positive integers X? For example, if X = {1,
// 3, 5}, you could climb 1, 3, or 5 steps at a time.

// Time: O(n * a)
// Space: O(n), could be reduced to O(a) by shrinking 'r' and rotating it.
int fn(const int n, const std::unordered_set<int>& A) {
  auto r = std::vector<int>(n, 0);
  r[0] = 1;
  r[1] = 1;

  for (int i = 0; i < n; ++i) {
    for (const auto& a : A) {
      r[i] += i - a >= 0 ? r[i - a] : 0;
    }
  }

  return r[n - 1];
}

TEST(StaircaseNCounts, Five) { EXPECT_EQ(fn(5, {1, 2}), 8); }

TEST(StaircaseNCounts, FiveThree) { EXPECT_EQ(fn(5, {1, 3, 5}), 5); }

TEST_MAIN();
