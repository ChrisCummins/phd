/*
 * Linked Lists. Taken from: http://maxnoy.com/interviews.html
 *
 * This is an extremely popular topic. I've had linked lists on every interview.
 * You must be able to produce simple clean linked list implementations quickly.
 *
 * Implement Insert and Delete for
 *   singly-linked linked list
 *   sorted linked list
 *   circular linked list
 *
 * - int Insert(node** head, int data)
 * - int Delete(node** head, int deleteMe)
 * - Split a linked list given a pivot value
 * - void Split(node* head, int pivot, node** lt, node** gt)
 * - Find if a linked list has a cycle in it. Now do it without
 *   marking nodes.
 * - Find the middle of a linked list. Now do it while only going
 *   through the list once. (same solution as finding cycles)
 */
#include <stdexcept>
#include <utility>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated"
#pragma GCC diagnostic ignored "-Wmissing-noreturn"
#pragma GCC diagnostic ignored "-Wpadded"
#pragma GCC diagnostic ignored "-Wshift-sign-overflow"
#pragma GCC diagnostic ignored "-Wundef"
#pragma GCC diagnostic ignored "-Wused-but-marked-unused"
#pragma GCC diagnostic ignored "-Wweak-vtables"
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpadded"

/////////////////////////
// Singly linked list: //
/////////////////////////

//
// A pretty opaque singly linked list implementation. A private
// subclass represents nodes. Manages node allocation dynamically.
//
template <typename T>
class SinglyLinkedList {
 public:
  SinglyLinkedList() : _head(nullptr) {}

  explicit SinglyLinkedList(std::initializer_list<T> il) : SinglyLinkedList() {
    auto rit = il.end() - 1;
    while (rit != il.begin() - 1) {
      push_front(*rit--);
    }
  }

  ~SinglyLinkedList() {
    node* n = _head;

    while (n) {
      auto tmp = n->next;
      delete n;
      n = tmp;
    }
  }

  void push_front(const T& data) {
    auto new_head = new node(data, _head);
    _head = new_head;
  }

  void erase_at(const std::size_t index) {
    if (index) {
      node* prev = _node_at(index - 1);
      node* next = prev->next->next;

      delete _node_at(index);
      prev->next = next;
    } else {
      // Removing the head:
      auto tmp = _head;
      _head = _head->next;
      delete tmp;
    }
  }

  void insert_after(const std::size_t index, const T& data) {
    node* current = _node_at(index);
    node* n = new node(data, current->next);
    current->next = n;
  }

  void insert_after(const std::size_t index, const T&& data) {
    node* current = _node_at(index);
    node* n = new node(std::move(data), current->next);
    current->next = n;
  }

  auto empty() const { return !_head; }

  auto& operator[](const std::size_t index) { return _node_at(index)->data; }

  auto& front() {
    if (empty()) throw std::out_of_range("SinglyLinkedList.front()");
    return _head->data;
  }

  template <typename _T>
  friend bool operator==(const SinglyLinkedList<_T>& lhs,
                         const SinglyLinkedList<_T>& rhs);

 private:
  class node;

  node* _head;

  auto _node_at(const std::size_t index) {
    node* n = _head;

    for (std::size_t i = 0; i < index; i++) {
      if (!n->next) throw std::out_of_range("SinglyLinkedList._node_at()");
      n = n->next;
    }

    return n;
  }

  class node {
   public:
    node(const T& _data, node* _next) : next(_next), data(_data) {}

    node* next;
    const T data;
  };
};

template <typename T>
bool operator==(const SinglyLinkedList<T>& lhs,
                const SinglyLinkedList<T>& rhs) {
  auto lit = lhs._head;
  auto rit = rhs._head;

  while (lit && rit) {
    if (lit->data != rit->data) return false;

    lit = lit->next;
    rit = rit->next;
  }

  return !(lit || rit);
}

TEST(SinglyLinkedList, push_front) {
  SinglyLinkedList<int> li;

  li.push_front(5);
  li.push_front(4);
  li.push_front(3);
  li.push_front(2);
  li.push_front(1);

  ASSERT_EQ(1, li.front());
  ASSERT_EQ(2, li[1]);
  ASSERT_EQ(3, li[2]);
  ASSERT_EQ(4, li[3]);
  ASSERT_EQ(5, li[4]);
}

TEST(SinglyLinkedList, initialiazer_list) {
  SinglyLinkedList<int> li{1, 2, 3, 4, 5};

  ASSERT_EQ(1, li.front());
  ASSERT_EQ(2, li[1]);
  ASSERT_EQ(3, li[2]);
  ASSERT_EQ(4, li[3]);
  ASSERT_EQ(5, li[4]);
}

TEST(SinglyLinkedList, relational_op) {
  SinglyLinkedList<int> l1{1, 2, 3, 4, 5};
  SinglyLinkedList<int> l2{1, 2, 3, 4, 5};
  SinglyLinkedList<int> l3{1, 6, 3, 4, 5};
  SinglyLinkedList<int> l4{1, 2, 3, 4, 5, 6};

  ASSERT_TRUE(l1 == l2);
  ASSERT_TRUE(l1 == l1);
  ASSERT_FALSE(l1 == l3);
  ASSERT_FALSE(l1 == l4);
}

TEST(SinglyLinkedList, insert_after) {
  SinglyLinkedList<int> li;

  li.push_front(1);
  li.insert_after(0, 2);
  li.insert_after(1, 4);
  li.insert_after(1, 3);
  li.insert_after(3, 5);

  ASSERT_EQ(1, li.front());
  ASSERT_EQ(2, li[1]);
  ASSERT_EQ(3, li[2]);
  ASSERT_EQ(4, li[3]);
  ASSERT_EQ(5, li[4]);
}

TEST(SinglyLinkedList, erase_at) {
  SinglyLinkedList<int> l1{1, 2, 2, 3, 4, 5, 6, 7};
  SinglyLinkedList<int> l2{1, 2, 3, 4, 5, 6, 7};
  SinglyLinkedList<int> l3{1, 3, 4, 5, 6, 7};
  SinglyLinkedList<int> l4{3, 4, 5, 6, 7};
  SinglyLinkedList<int> l5{3, 4, 5, 6};

  l1.erase_at(1);
  ASSERT_TRUE(l1 == l2);
  l1.erase_at(1);
  ASSERT_TRUE(l1 == l3);
  l1.erase_at(0);
  ASSERT_TRUE(l1 == l4);
  l1.erase_at(4);
  ASSERT_TRUE(l1 == l5);
}

/////////////////////////
// Sorted linked list: //
/////////////////////////

template <typename T>
class SortedLinkedList {
 public:
  SortedLinkedList() : _head(nullptr) {}

  explicit SortedLinkedList(std::initializer_list<T> il) : SortedLinkedList() {
    auto rit = il.end() - 1;
    while (rit != il.begin() - 1) {
      insert(*rit--);
    }
  }

  ~SortedLinkedList() {
    auto n = _head;

    while (n) {
      auto tmp = n->next;
      delete n;
      n = tmp;
    }
  }

  void insert(const T& data) {
    node *prev = nullptr, *curr = _head;

    // Iterate to correct point in list.
    while (curr && curr->data < data) {
      prev = curr;
      curr = curr->next;
    }

    auto newnode = new class node(data, curr);

    if (prev)
      prev->next = newnode;
    else
      _head = newnode;
  }

  void erase(const T& data) {
    node *prev = nullptr, *curr = _head;

    while (curr && curr->data != data) {
      prev = curr;
      curr = curr->next;
    }

    // We reached the end of the list and didn't find the value:
    if (!curr) throw std::invalid_argument("SortedLinkedList.erase()");

    if (prev)
      prev->next = curr->next;
    else
      _head = curr->next;

    delete curr;
  }

  auto empty() const { return !_head; }

  auto& operator[](const std::size_t index) { return _node_at(index)->data; }

  auto& front() {
    if (empty()) throw std::out_of_range("SortedLinkedList.front()");
    return _head->data;
  }

  template <typename _T>
  friend bool operator==(const SortedLinkedList<_T>& lhs,
                         const SortedLinkedList<_T>& rhs);

 private:
  class node;

  node* _head;

  auto _node_at(const std::size_t index) {
    node* n = _head;

    for (std::size_t i = 0; i < index; i++) {
      if (!n->next) throw std::out_of_range("SortedLinkedList._node_at()");
      n = n->next;
    }

    return n;
  }

  class node {
   public:
    node(const T& _data, node* _next) : next(_next), data(_data) {}

    node* next;
    const T data;
  };
};

template <typename T>
bool operator==(const SortedLinkedList<T>& lhs,
                const SortedLinkedList<T>& rhs) {
  auto lit = lhs._head;
  auto rit = rhs._head;

  while (lit && rit) {
    if (lit->data != rit->data) return false;

    lit = lit->next;
    rit = rit->next;
  }

  return !(lit || rit);
}

TEST(SortedLinkedList, insert) {
  SortedLinkedList<int> li;

  li.insert(5);
  li.insert(3);
  li.insert(4);
  li.insert(1);
  li.insert(2);

  ASSERT_EQ(1, li.front());
  ASSERT_EQ(2, li[1]);
  ASSERT_EQ(3, li[2]);
  ASSERT_EQ(4, li[3]);
  ASSERT_EQ(5, li[4]);
}

TEST(SortedLinkedList, initialiazer_list) {
  SortedLinkedList<int> li{4, 3, 2, 1, 5};

  ASSERT_EQ(1, li.front());
  ASSERT_EQ(2, li[1]);
  ASSERT_EQ(3, li[2]);
  ASSERT_EQ(4, li[3]);
  ASSERT_EQ(5, li[4]);
}

TEST(SortedLinkedList, relational_op) {
  SortedLinkedList<int> l1{1, 2, 3, 4, 5};
  SortedLinkedList<int> l2{1, 2, 3, 4, 5};
  SortedLinkedList<int> l3{1, 6, 3, 4, 5};
  SortedLinkedList<int> l4{1, 2, 3, 4, 5, 6};

  ASSERT_TRUE(l1 == l2);
  ASSERT_TRUE(l1 == l1);
  ASSERT_FALSE(l1 == l3);
  ASSERT_FALSE(l1 == l4);
}

TEST(SortedLinkedList, erase) {
  SortedLinkedList<int> l1{1, 2, 2, 3, 4, 5, 6, 7};
  SortedLinkedList<int> l2{1, 2, 3, 4, 5, 6, 7};
  SortedLinkedList<int> l3{1, 3, 4, 5, 6, 7};
  SortedLinkedList<int> l4{3, 4, 5, 6, 7};
  SortedLinkedList<int> l5{3, 4, 5, 6};

  l1.erase(2);
  ASSERT_TRUE(l1 == l2);
  l1.erase(2);
  ASSERT_TRUE(l1 == l3);
  l1.erase(1);
  ASSERT_TRUE(l1 == l4);
  l1.erase(7);
  ASSERT_TRUE(l1 == l5);
}

#pragma GCC diagnostic pop  // -Wpadded

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
