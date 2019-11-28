#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated"
#pragma GCC diagnostic ignored "-Wmissing-noreturn"
#pragma GCC diagnostic ignored "-Wpadded"
#pragma GCC diagnostic ignored "-Wshift-sign-overflow"
#pragma GCC diagnostic ignored "-Wundef"
#pragma GCC diagnostic ignored "-Wused-but-marked-unused"
#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

#include <iostream>

//
// Doubly-linked list with constant time O(1) reverse method. Uses a
// direction bit.
//
template <typename T>
class quick_reverse_list {
 private:
  class __node;
  class __iterator;

 public:
  enum direction { forward = 0, backward = 1 };

  quick_reverse_list() {}

  quick_reverse_list(std::initializer_list<T> il,
                     direction direction = direction::forward)
      : _direction(direction) {
    for (auto val : il) push_back(val);
  }

  ~quick_reverse_list() {
    auto* n = _head;
    while (n) {
      auto* tmp = n->forward();
      delete n;
      n = tmp;
    }
  }

  void push_back(const T& data) {
    if (_head) {
      auto* newnode = new __node{data, nullptr, _tail};
      _tail->forward() = newnode;
      _tail = newnode;
    } else {
      _head = new __node{data};
      _tail = _head;
    }
  }

  void reverse() {
    if (_direction == direction::forward)
      _direction = direction::backward;
    else
      _direction = direction::forward;
  }

  const __iterator begin() const {
    if (_direction == direction::forward)
      return __iterator(_head, _direction);
    else
      return __iterator(_tail, _direction);
  }

  const __iterator end() const { return __iterator(nullptr, _direction); }

  friend std::ostream& operator<<(std::ostream& out,
                                  const quick_reverse_list& list) {
    for (auto& val : list) out << val << ' ';
    return out;
  }

 private:
  //
  // Node.
  //
  class __node {
   public:
    explicit __node(const T& data, __node* forward = nullptr,
                    __node* backward = nullptr)
        : _data(data), _forward(forward), _backward(backward) {}
    explicit __node(T&& data, __node* forward = nullptr,
                    __node* backward = nullptr)
        : _data(std::move(data)), _forward(forward), _backward(backward) {}
    T& data() { return _data; }
    const T& data() const { return _data; }
    __node*& forward() { return _forward; }
    __node*& backward() { return _backward; }
    friend std::ostream& operator<<(std::ostream& out, const __node& n) {
      out << n._data;
      return out;
    }

   private:
    T _data;
    __node* _forward;
    __node* _backward;
  };

  //
  // Iterator.
  //
  class __iterator {
   public:
    __iterator(__node* node, const direction direction)
        : _node(node), _direction(direction) {}
    T& operator*() { return _node->data(); }
    __iterator& operator++() {
      if (_direction == direction::forward)
        _node = _node->forward();
      else
        _node = _node->backward();
      return *this;
    }
    __iterator& operator--() {
      if (_direction == direction::forward)
        _node = _node->backward();
      else
        _node = _node->forward();
      return *this;
    }

    friend bool operator==(const __iterator& lhs, const __iterator& rhs) {
      return lhs._node == rhs._node;
    }

    friend bool operator!=(const __iterator& lhs, const __iterator& rhs) {
      return !(lhs == rhs);
    }

   private:
    __node* _node;
    const direction _direction;
  };

  enum direction _direction = direction::forward;
  __node* _head = nullptr;
  __node* _tail = nullptr;
};

int main() {
  quick_reverse_list<int> q{1, 2, 3, 4, 5};

  std::cout << q << std::endl;
  q.reverse();
  std::cout << q << std::endl;

  return 0;
}
