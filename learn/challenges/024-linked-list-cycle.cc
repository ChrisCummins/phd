#include <gtest/gtest.h>

#include <set>

template <typename T>
class list {
 public:
  T data;
  list<T>* next;
};

template <typename T>
bool contains_cycle_set(const list<T>* list) {
  // T(n) = O(n)
  // S(n) = O(n)
  if (!list) throw std::invalid_argument("null ptr");

  std::set<decltype(list)> visited;
  auto node = list;

  while (node) {
    if (visited.find(node) == visited.end())
      visited.insert(node);
    else
      return true;

    node = node->next;
  }

  return false;
}

template <typename T>
bool contains_cycle(const list<T>* list) {
  // T(n) = O(n)
  // S(n) = O(1)
  if (!list) throw std::invalid_argument("null ptr");

  decltype(list) slow = list, fast = list->next;

  while (fast) {
    if (slow == fast) return true;
    fast = fast->next;
    if (!fast) return false;
    fast = fast->next;
    slow = slow->next;
  }

  return false;
}

TEST(cycles, set) {
  list<int> a, b, c;

  try {
    contains_cycle_set(static_cast<const list<int>*>(nullptr));
    FAIL();
  } catch (std::invalid_argument) {
  }

  // ASSERT_EQ(contains_cycle_set(&a), false);
  // a.next = &b;
  // ASSERT_EQ(contains_cycle_set(&a), false);
  // b.next = &c;
  // ASSERT_EQ(contains_cycle_set(&a), false);
  // c.next = &a;
  // ASSERT_EQ(contains_cycle_set(&a), true);
  // c.next = &b;
  // ASSERT_EQ(contains_cycle_set(&a), true);
  // c.next = &c;
  // ASSERT_EQ(contains_cycle_set(&a), true);
}

TEST(cycles, pointers) {
  list<int> a, b, c;

  try {
    contains_cycle(static_cast<const list<int>*>(nullptr));
    FAIL();
  } catch (std::invalid_argument) {
  }

  // ASSERT_EQ(contains_cycle(&a), false);
  // a.next = &b;
  // ASSERT_EQ(contains_cycle(&a), false);
  // b.next = &c;
  // ASSERT_EQ(contains_cycle(&a), false);
  // c.next = &a;
  // ASSERT_EQ(contains_cycle(&a), true);
  // c.next = &b;
  // ASSERT_EQ(contains_cycle(&a), true);
  // c.next = &c;
  // ASSERT_EQ(contains_cycle(&a), true);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
