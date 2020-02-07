// Find the kth highest number in a binary tree.
//
// Note: assume binary search tree (i.e. complete, ordered).
#include <stack>
#include <vector>

#include "labm8/cpp/test.h"

using std::stack;
using std::vector;

// Use flattened node array for BST, where 0 means no element.
// Helper functions:

bool Empty(const vector<int>& bst, int i) { return bst[i] == 0; }

template <typename T>
int Child(const vector<T>& bst, int i, bool right) {
  // Invalid root.
  if (i < 0 || i >= bst.size() || Empty(bst, i)) {
    return 0;
  }
  int n = (i * 2) + 1;
  if (right) {
    ++n;
  }
  // Invalid child.
  if (n >= bst.size() || Empty(bst, n)) {
    return 0;
  }
  return n;
}

template <typename T>
int Left(const vector<T>& bst, int i) {
  return Child(bst, i, /*right=*/false);
}

template <typename T>
int Right(const vector<T>& bst, int i) {
  return Child(bst, i, /*right=*/true);
}

// DFS in-order traversal (right then left), stop when stack size reaches k.
template <typename T>
void KthHighestValue(const vector<T>& bst, int i, int* k, T* val) {
  int r = Right(bst, i);
  int l = Left(bst, i);

  if (r) {
    KthHighestValue(bst, r, k, val);
    if (*k < 0) {
      return;
    }
  }
  *k = *k - 1;
  if (*k < 0) {
    *val = bst[i];
    return;
  }

  if (l) {
    KthHighestValue(bst, l, k, val);
  }
}

// Time: O(n)
// Space: O(log n)
template <typename T>
T KthHighestValue(const vector<T>& bst, int k) {
  T ret(0);
  if (!bst.size()) {
    return ret;
  }

  KthHighestValue(bst, 0, &k, &ret);

  return ret;
}

TEST(KthHighestValue, EmptyTree) {
  EXPECT_EQ(KthHighestValue(vector<int>({}), 0), 0);
}

TEST(KthHighestValue, ExampleTree) {
  //     10
  //    / \
  //   5   15
  //  / \
  // 3   7
  const vector<int> t{10, 5, 15, 3, 7, 0, 0};
  EXPECT_EQ(KthHighestValue(t, 0), 15);
  EXPECT_EQ(KthHighestValue(t, 1), 10);
  EXPECT_EQ(KthHighestValue(t, 2), 7);
  EXPECT_EQ(KthHighestValue(t, 3), 5);
  EXPECT_EQ(KthHighestValue(t, 4), 3);
  EXPECT_EQ(KthHighestValue(t, 5), 0);
  EXPECT_EQ(KthHighestValue(t, 6), 0);
}

TEST_MAIN();
