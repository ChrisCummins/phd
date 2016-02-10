/*
 * Implement a function to check if a linked list is a palindrome.
 */
#include "./ctci.h"

#include <list>

//
// Solution for doubly linked list. Position forward and reverse
// iterators at either end and advance towards midpoint.
//
template<typename T>
bool is_palindrone(const std::list<T>& list) {
  auto left = list.begin();
  auto right = list.rbegin();

  for (size_t i = 0; i < list.size() / 2; i++)
    if (*left++ != *right++)
      return false;

  return true;
}

TEST(ListPalindrone, is_palindrone) {
  const std::list<int> l1{1, 2, 3, 4, 5};
  ASSERT_FALSE(is_palindrone(l1));

  const std::list<int> l2{1, 2, 3, 3, 2, 1};
  ASSERT_TRUE(is_palindrone(l2));

  const std::list<int> l3{1, 2, 3, 2, 1};
  ASSERT_TRUE(is_palindrone(l3));
}

CTCI_MAIN();
