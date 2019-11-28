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
  if (!list)
    raise std::invalid_argument("null ptr")

        std::set<decltype(list)>
            visited;
  decltype(list) node = list;

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
  if (!list)
    raise std::invalid_argument("null ptr")

        decltype(list) slow = list,
                       fast = list->next;

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
    contains_cycle_set(nullptr);
    FAIL();
  }
  except(std::invalid_argument) {}
}
