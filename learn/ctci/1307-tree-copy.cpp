/*
 * Write a method that takes a pointer to a Node structure as a
 * parameter and returns a complete copy of the passed in data
 * structure. The Node data structure contains two pointers to other
 * Nodes.
 */
#include <array>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpadded"
#pragma GCC diagnostic ignored "-Wundef"
#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

class Node {
 public:
  Node *left;
  Node *right;
  int data;
  int _pad;

  explicit Node(int val) : left(nullptr), right(nullptr), data(val) {}
  ~Node() {}
};

Node *copyNodes1(Node *node) {
  auto* newNode = new Node(node->data);

  if (node->left)
    newNode->left = copyNodes1(node->left);
  if (node->right)
    newNode->right = copyNodes1(node->right);

  return newNode;
}

// Unit tests

TEST(TreeCopy, tests) {
  std::array<Node*, 6> nodes = {
    new Node(0),
    new Node(1),
    new Node(2),
    new Node(3),
    new Node(4),
    new Node(5)
  };

  // Assemble tree:
  nodes[0]->left = nodes[1];
  nodes[0]->right = nodes[2];
  nodes[1]->left = nodes[3];
  nodes[3]->left = nodes[4];
  nodes[2]->left = nodes[5];

  auto* _newnodes = copyNodes1(nodes[0]);

  std::array<Node*, 6> newnodes = {
    _newnodes,
    _newnodes->left,
    _newnodes->right,
    _newnodes->left->left,
    _newnodes->left->left->left,
    _newnodes->right->left
  };

  for (size_t i = 0; i < nodes.size(); i++) {
    // Addresses don't match (i.e. they're different objects).
    ASSERT_NE(nodes[i], newnodes[i]);
    // Data does match.
    ASSERT_EQ(nodes[i]->data, newnodes[i]->data);

    if (nodes[i]->left)
      ASSERT_EQ(nodes[i]->left->data, newnodes[i]->left->data);
    if (nodes[i]->right)
      ASSERT_EQ(nodes[i]->right->data, newnodes[i]->right->data);
  }

  for (auto* n : nodes)
    delete n;
  for (auto *n : newnodes)
    delete n;
}


int main(int argc, char **argv) {
  // Run unit tests:
  testing::InitGoogleTest(&argc, argv);
  const auto ret = RUN_ALL_TESTS();

  // Run benchmarks:
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();

  return ret;
}
