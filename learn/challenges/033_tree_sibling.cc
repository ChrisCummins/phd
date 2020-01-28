// We have a tree: {left, right sibling}.
//
// You get the root of the tree which has connections populated and we want to
// compute the sibling one.
#include "labm8/cpp/test.h"

#include <queue>
#include <utility>

using std::pair;
using std::queue;

struct Node {
  Node* left = nullptr;
  Node* right = nullptr;
  Node* sibling = nullptr;
};

// Time: O(n)
// Space: O(n)
void AddTreeSiblings(Node* root) {
  queue<pair<Node*, int>> q;
  q.push({root, 0});
  int level = 0;
  Node* sibling = nullptr;

  while (!q.empty()) {
    auto& x = q.front();
    Node* n = x.first;
    int nl = x.second;
    if (level == nl) n->sibling = sibling;
    level = nl;
    sibling = n;
    q.pop();
    if (n->right) {
      q.push({n->right, level + 1});
    }
    if (n->left) {
      q.push({n->left, level + 1});
    }
  }
}

TEST(AddTreeSiblings, OnlyRoot) {
  Node root;

  AddTreeSiblings(&root);

  EXPECT_EQ(root.sibling, nullptr);
  EXPECT_EQ(root.left, nullptr);
  EXPECT_EQ(root.right, nullptr);
}

TEST(AddTreeSiblings, OnlyLeft) {
  Node root;
  Node left;
  root.left = &left;

  AddTreeSiblings(&root);

  EXPECT_EQ(root.sibling, nullptr);
  EXPECT_EQ(root.left, &left);
  EXPECT_EQ(root.right, nullptr);

  EXPECT_EQ(left.sibling, nullptr);
  EXPECT_EQ(left.sibling, nullptr);
  EXPECT_EQ(left.sibling, nullptr);
}

TEST(AddTreeSiblings, OnlyRight) {
  Node root;
  Node right;
  root.right = &right;

  AddTreeSiblings(&root);

  EXPECT_EQ(root.sibling, nullptr);
  EXPECT_EQ(root.left, nullptr);
  EXPECT_EQ(root.right, &right);

  EXPECT_EQ(right.sibling, nullptr);
  EXPECT_EQ(right.sibling, nullptr);
  EXPECT_EQ(right.sibling, nullptr);
}

TEST(AddTreeSiblings, CompleteTreeDepthOne) {
  Node root;
  Node left;
  Node right;
  root.left = &left;
  root.right = &right;

  AddTreeSiblings(&root);

  EXPECT_EQ(root.sibling, nullptr);
  EXPECT_EQ(root.left, &left);
  EXPECT_EQ(root.right, &right);

  EXPECT_EQ(left.sibling, &right);
  EXPECT_EQ(left.left, nullptr);
  EXPECT_EQ(left.right, nullptr);

  EXPECT_EQ(right.sibling, nullptr);
  EXPECT_EQ(right.left, nullptr);
  EXPECT_EQ(right.right, nullptr);
}

TEST(AddTreeSiblings, BiggerTree) {
  //        root
  //       /    \
  //      A ---- B
  //     / \      \
  //    C - D ---- E
  //   / \   \    / \
  //  F - G - H -I - J
  //   \
  //    K
  Node root;
  Node A;
  Node B;
  Node C;
  Node D;
  Node E;
  Node F;
  Node G;
  Node H;
  Node I;
  Node J;
  Node K;

  root.left = &A;
  root.right = &B;

  A.left = &C;
  A.right = &D;
  B.right = &E;

  C.left = &F;
  C.right = &G;
  D.right = &H;
  E.left = &I;
  E.right = &J;

  F.right = &K;

  AddTreeSiblings(&root);

  EXPECT_EQ(root.sibling, nullptr);
  EXPECT_EQ(root.left, &A);
  EXPECT_EQ(root.right, &B);

  EXPECT_EQ(A.sibling, &B);
  EXPECT_EQ(A.left, &C);
  EXPECT_EQ(A.right, &D);
  EXPECT_EQ(B.sibling, nullptr);
  EXPECT_EQ(B.left, nullptr);
  EXPECT_EQ(B.right, &E);

  EXPECT_EQ(C.sibling, &D);
  EXPECT_EQ(C.left, &F);
  EXPECT_EQ(C.right, &G);
  EXPECT_EQ(D.sibling, &E);
  EXPECT_EQ(D.left, nullptr);
  EXPECT_EQ(D.right, &H);
  EXPECT_EQ(E.sibling, nullptr);
  EXPECT_EQ(E.left, &I);
  EXPECT_EQ(E.right, &J);

  EXPECT_EQ(F.sibling, &G);
  EXPECT_EQ(F.left, nullptr);
  EXPECT_EQ(F.right, &K);
  EXPECT_EQ(G.sibling, &H);
  EXPECT_EQ(G.left, nullptr);
  EXPECT_EQ(G.right, nullptr);
  EXPECT_EQ(H.sibling, &I);
  EXPECT_EQ(H.left, nullptr);
  EXPECT_EQ(H.right, nullptr);
  EXPECT_EQ(J.sibling, nullptr);
  EXPECT_EQ(J.left, nullptr);
  EXPECT_EQ(J.right, nullptr);

  EXPECT_EQ(K.sibling, nullptr);
  EXPECT_EQ(K.left, nullptr);
  EXPECT_EQ(K.right, nullptr);
}

TEST_MAIN();
