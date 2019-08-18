// This problem was asked by Google.
//
// A unival tree (which stands for "universal value") is a tree where all nodes
// under it have the same value.
//
// Given the root to a binary tree, count the number of unival subtrees.
//
// For example, the following tree has 5 universal subtrees:
//
//    0
//   / \
//  1   0
//     / \
//    1   0
//   / \
//  1   1
#include <iostream>
#include <set>
#include <stack>
#include <tuple>

template <typename T>
class UniversalTree {
 public:
  UniversalTree(const T& value)
      : value_(value), left_(nullptr), right_(nullptr) {}

  const T value_;
  UniversalTree* left_;
  UniversalTree* right_;

  // Recursive solution. Tracks universal subtree count, and returns whether
  // the current tree is a universal subtree (if any).
  //
  // Time: O(n), visit every node once.
  // Space: O(n), call stack is depth of the tree.
  bool IsUniversalSubtree(int* count) const {
    bool is_universal_subtree = true;

    for (const UniversalTree* child : {left_, right_}) {
      if (child) {
        auto childval = child->IsUniversalSubtree(count);
        is_universal_subtree &= childval && child->value_ == value_;
      }
    }

    if (is_universal_subtree) {
      *count += 1;
    }

    return is_universal_subtree;
  }

  int GetUniversalSubtreeCount() const {
    int count = 0;
    IsUniversalSubtree(&count);
    return count;
  }
};

void Test1() {
  UniversalTree<int> a(0);
  UniversalTree<int> b(1);
  UniversalTree<int> c(0);
  UniversalTree<int> d(1);
  UniversalTree<int> e(1);
  UniversalTree<int> f(1);
  UniversalTree<int> g(0);

  a.left_ = &b;
  a.right_ = &c;
  c.left_ = &d;
  d.left_ = &e;
  d.right_ = &f;
  c.right_ = &g;

  std::cout << "a = " << a.GetUniversalSubtreeCount() << std::endl;
  std::cout << "b = " << b.GetUniversalSubtreeCount() << std::endl;
  std::cout << "c = " << c.GetUniversalSubtreeCount() << std::endl;
  std::cout << "d = " << d.GetUniversalSubtreeCount() << std::endl;
  std::cout << "e = " << e.GetUniversalSubtreeCount() << std::endl;
  std::cout << "f = " << f.GetUniversalSubtreeCount() << std::endl;
  std::cout << "g = " << g.GetUniversalSubtreeCount() << std::endl;
}

void Test2() {
  UniversalTree<int> a(0);
  UniversalTree<int> b(1);
  UniversalTree<int> c(0);

  a.left_ = &b;
  a.right_ = &c;

  std::cout << "a = " << a.GetUniversalSubtreeCount() << std::endl;
  std::cout << "b = " << b.GetUniversalSubtreeCount() << std::endl;
  std::cout << "c = " << c.GetUniversalSubtreeCount() << std::endl;
}

void Test3() {
  UniversalTree<int> a(1);
  UniversalTree<int> b(1);
  UniversalTree<int> c(1);

  a.left_ = &b;
  a.right_ = &c;

  std::cout << "a = " << a.GetUniversalSubtreeCount() << std::endl;
  std::cout << "b = " << b.GetUniversalSubtreeCount() << std::endl;
  std::cout << "c = " << c.GetUniversalSubtreeCount() << std::endl;
}

int main(int argc, char** argv) {
  std::cout << "Test 1\n";
  Test1();
  std::cout << "Test 2\n";
  Test2();
  std::cout << "Test 3\n";
  Test3();
  return 0;
}
