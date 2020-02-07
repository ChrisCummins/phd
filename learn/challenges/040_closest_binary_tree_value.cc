// Write a function that returns the closest value from a BST.
// Input value is float, tree is ints.
#include "labm8/cpp/test.h"

#include <cmath>
#include <iostream>
#include <vector>

using std::numeric_limits;
using std::vector;

// Helper functions for walking the tree. The BST is implemented as an array,
// with the root at index 1 (index 0 is unoccupied).
template <typename T>
bool InBounds(const vector<T>& tree, size_t i) {
  return i < tree.size() && tree[i];
}

template <typename T>
size_t Left(const vector<T>& tree, size_t i) {
  return InBounds(tree, i * 2) ? i * 2 : 0;
}

template <typename T>
size_t Right(const vector<T>& tree, size_t i) {
  return InBounds(tree, i * 2 + 1) ? i * 2 + 1 : 0;
}

// Time: O(h) - height of the tree, O(log n) if balanced
// Space: O(h) - height of the tree, O(log n) if balanced
void FN(const vector<int>& T, float x, size_t i, float* bestDistance,
        int* bestVal) {
  // out of bounds. only occurs with an empty tree.
  if (!InBounds(T, i)) {
    return;
  }

  float delta = abs(x - T[i]);
  if (delta < *bestDistance) {
    *bestDistance = delta;
    *bestVal = T[i];
  }

  if (T[i] > x && Left(T, i)) {
    FN(T, x, Left(T, i), bestDistance, bestVal);
  } else if (T[i] < x && Right(T, i)) {
    FN(T, x, Right(T, i), bestDistance, bestVal);
  }
}

int FN(const vector<int>& T, float x) {
  int bestVal = -1;
  float bestDistance = numeric_limits<float>::infinity();

  FN(T, x, 1, &bestDistance, &bestVal);
  return bestVal;
}

TEST(ClosestBinarySearchTreeValue, EmptyTree) {
  EXPECT_EQ(FN({}, 5), -1);
  EXPECT_EQ(FN({0}, 5), -1);
  EXPECT_EQ(FN({0, 0}, 5), -1);
}

TEST(ClosestBinarySearchTreeValue, SingleElementExactMatch) {
  EXPECT_EQ(FN({0, 5}, 5), 5);
}

TEST(ClosestBinarySearchTreeValue, SingleElementClosestMatch) {
  EXPECT_EQ(FN({0, 3}, 5), 3);
}

TEST(ClosestBinarySearchTreeValue, ExampleTree) {
  //     5
  //    / \
	//   3   7
  //  / \   \
	// 1   4   10
  const vector<int> tree{0, 5, 3, 7, 1, 4, 0, 10};

  EXPECT_EQ(FN({0, 5, 3, 7, 1, 4, 0, 10}, 7.5), 7);
  EXPECT_EQ(FN({0, 5, 3, 7, 1, 4, 0, 10}, 8.6), 10);
  EXPECT_EQ(FN({0, 5, 3, 7, 1, 4, 0, 10}, 100), 10);
  EXPECT_EQ(FN({0, 5, 3, 7, 1, 4, 0, 10}, 0), 1);
  EXPECT_EQ(FN({0, 5, 3, 7, 1, 4, 0, 10}, 3), 3);
  EXPECT_EQ(FN({0, 5, 3, 7, 1, 4, 0, 10}, 3.9), 4);
  EXPECT_EQ(FN({0, 5, 3, 7, 1, 4, 0, 10}, 4.1), 4);
  EXPECT_EQ(FN({0, 5, 3, 7, 1, 4, 0, 10}, 5.1), 5);
}

TEST_MAIN();
