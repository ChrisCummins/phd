/*
 * Implement an algorithm to find the kth to last element of a single liked
 * list.
 */
#include "./ctci.h"

#include <forward_list>
#include <vector>

//
// Store a circular buffer of the last k+1 elements.
//
// O(n) time, O(n) space.
//
template <typename T>
typename std::forward_list<T>::const_iterator k_last_elem(
    const std::forward_list<T> lst, const size_t k) {
  std::vector<typename std::forward_list<T>::const_iterator> tmp(k + 1);
  size_t i = 0;
  auto first = lst.begin();

  while (first != lst.end()) {
    tmp[i] = first++;
    i = (i + 1) % (k + 1);
  }

  return tmp[i];
}

///////////
// Tests //
///////////

TEST(challenge, basic) {
  std::forward_list<int> l{1, 2, 3, 4, 5};

  ASSERT_EQ(4, *k_last_elem(l, 1));
  ASSERT_EQ(3, *k_last_elem(l, 2));
  ASSERT_EQ(2, *k_last_elem(l, 3));
  ASSERT_EQ(1, *k_last_elem(l, 4));
}

CTCI_MAIN();
