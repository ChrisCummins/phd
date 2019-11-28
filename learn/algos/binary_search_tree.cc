#include <algorithm>
#include <iostream>

template <typename T>
class BST {
 public:
  T data;
  BST *left;
  BST *right;

  BST(const T &val, BST *const left = nullptr, BST *const right = nullptr)
      : data(val), left(left), right(right) {}

  void insert(const T &val) {
    if (val > data) {
      if (right) {
        right->insert(val);
      } else {
        right = new BST(val);
      }
    } else {
      if (left) {
        left->insert(val);
      } else {
        left = new BST(val);
      }
    }
  }

  friend std::ostream &operator<<(std::ostream &o, const BST &t) {
    o << t.data << " ";
    return o;
  }

  int height() { return _height(1); }

 private:
  int _height(int h) {
    int hleft = h;
    int hright = h;

    if (left) {
      hleft = left->_height(h + 1);
    }
    if (right) {
      hright = right->_height(h + 1);
    }

    return std::max<int>(hleft, hright);
  }
};

template <typename T>
void inorder_traverse(const BST<T> *const root) {
  if (root) {
    if (root->left) {
      std::cout << "(";
      inorder_traverse(root->left);
      std::cout << ")";
    }
    std::cout << root->data;
    if (root->right) {
      std::cout << "(";
      inorder_traverse(root->right);
      std::cout << ")";
    }
  }
}

template <typename T>
void preorder_traverse(const BST<T> *const root) {
  if (root) {
    std::cout << root->data;
    if (root->left || root->right) std::cout << "(";
    if (root->left) preorder_traverse(root->left);
    if (root->right) preorder_traverse(root->right);
    if (root->left || root->right) std::cout << ")";
  }
}

template <typename T>
void invert(BST<T> *const root) {
  if (root->left) invert(root->left);
  if (root->right) invert(root->right);

  std::swap(root->left, root->right);
}

int main(int argc, char **argv) {
  auto root = BST<int>(10);
  root.insert(5);
  root.insert(4);
  root.insert(3);
  root.insert(2);
  root.insert(1);
  root.insert(6);
  root.insert(7);
  root.insert(15);
  root.insert(30);
  root.insert(20);

  std::cout << "Height:   " << root.height() << std::endl;

  std::cout << "Inorder:  ";
  inorder_traverse(&root);
  std::cout << std::endl;

  std::cout << "Preorder: ";
  preorder_traverse(&root);
  std::cout << std::endl;

  invert(&root);
  std::cout << "Inverted: ";
  inorder_traverse(&root);
  std::cout << std::endl;
  invert(&root);
}
