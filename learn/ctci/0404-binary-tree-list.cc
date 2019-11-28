/*
 * Given a binary tree, design an algorithm which creates a linked
 * list of all the nodes at each depth (e.g. if you have a tree with
 * depth D, you'll have D linked lists).
 */
#include "./ctci.h"

#include <functional>
#include <list>
#include <vector>

template <typename T>
class node {
 public:
  explicit node(const T& _data, node* _left = nullptr, node* _right = nullptr)
      : data(_data), left(_left), right(_right) {}

  explicit node(const T&& _data, node* _left = nullptr, node* _right = nullptr)
      : data(std::move(_data)), left(_left), right(_right) {}

  T data;
  node* left;
  node* right;
};

template <typename T>
using tree_lists = std::vector<std::list<const node<T>*>>;

template <typename T>
tree_lists<T> tree_to_lists(const node<T>* const tree) {
  std::function<void(const node<T>* const tree, tree_lists<T>& out,
                     size_t depth)>
      _tree_to_lists = [&](const auto* root, auto& out, auto depth) {
        while (out.size() <= depth) out.emplace_back();
        out[depth].push_back(root);

        if (root->left) _tree_to_lists(root->left, out, depth + 1);
        if (root->right) _tree_to_lists(root->right, out, depth + 1);
      };

  tree_lists<T> lists;
  _tree_to_lists(tree, lists, 0);
  return lists;
}

///////////
// Tests //
///////////

TEST(BinaryTreeToList, tree_to_list) {
  node<int> _nodes[] = {
      node<int>(0), node<int>(1), node<int>(2),
      node<int>(3), node<int>(4), node<int>(5),
  };

  node<int>* const tree = _nodes;

  tree->left = &_nodes[1];
  tree->right = &_nodes[2];
  tree->left->left = &_nodes[3];
  tree->left->right = &_nodes[4];
  tree->right->left = &_nodes[5];

  auto lists = tree_to_lists(tree);

  ASSERT_EQ(3u, lists.size());

  ASSERT_EQ(1u, lists[0].size());
  ASSERT_EQ(2u, lists[1].size());
  ASSERT_EQ(3u, lists[2].size());
}

CTCI_MAIN();
