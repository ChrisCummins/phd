/*
 * Write a method that takes a pointer to a Node structure as a
 * parameter and returns a complete copy of the passed in data
 * structure. The Node data structure contains two pointers to other
 * Nodes.
 */
#include "./ctci.h"

#include <array>

class Node {
 public:
  Node *left;
  Node *right;
  int data;
  int _pad;

  explicit Node(int val) : left(nullptr), right(nullptr), data(val) {}
  ~Node() {}
};

//
// First solution. Allocate new node and recursively assign children.
//
// O(n) time, O(n) space.
//
Node *deep_copy(Node *node) {
  auto *newNode = new Node(node->data);

  if (node->left) newNode->left = deep_copy(node->left);
  if (node->right) newNode->right = deep_copy(node->right);

  return newNode;
}

///////////
// Tests //
///////////

TEST(TreeCopy, tests) {
  std::array<Node *, 6> nodes = {new Node(0), new Node(1), new Node(2),
                                 new Node(3), new Node(4), new Node(5)};

  // Assemble tree:
  nodes[0]->left = nodes[1];
  nodes[0]->right = nodes[2];
  nodes[1]->left = nodes[3];
  nodes[3]->left = nodes[4];
  nodes[2]->left = nodes[5];

  auto *_newnodes = deep_copy(nodes[0]);

  std::array<Node *, 6> newnodes = {_newnodes,
                                    _newnodes->left,
                                    _newnodes->right,
                                    _newnodes->left->left,
                                    _newnodes->left->left->left,
                                    _newnodes->right->left};

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

  for (auto *n : nodes) delete n;
  for (auto *n : newnodes) delete n;
}

CTCI_MAIN();
