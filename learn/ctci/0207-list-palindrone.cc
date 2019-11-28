/*
 * Implement a function to check if a linked list is a palindrome.
 */
#include "./ctci.h"

#include <list>

//
// Solution for doubly linked list. Position forward and reverse
// iterators at either end and advance towards midpoint.
//
template <typename T>
bool is_palindrone(const std::list<T>& list) {
  auto left = list.begin();
  auto right = list.rbegin();

  for (size_t i = 0; i < list.size() / 2; i++)
    if (*left++ != *right++) return false;

  return true;
}

///////////
// Tests //
///////////

TEST(ListPalindrone, is_palindrone) {
  const std::list<int> l1{1, 2, 3, 4, 5};
  ASSERT_FALSE(is_palindrone(l1));

  const std::list<int> l2{1, 2, 3, 3, 2, 1};
  ASSERT_TRUE(is_palindrone(l2));

  const std::list<int> l3{1, 2, 3, 2, 1};
  ASSERT_TRUE(is_palindrone(l3));
}

////////////////
// Benchmarks //
////////////////

static const size_t BM_length_min = 8;
static const size_t BM_length_max = 10 << 10;

void BM_is_palindrone(benchmark::State& state) {
  const auto len = static_cast<size_t>(state.range(0));
  const auto half = len / 2;

  std::list<int> list;

  // Create palindrone.
  for (int i = 0; i < static_cast<int>(half); i++) list.push_back(i);
  auto it = list.crbegin();
  while (it != list.crend()) list.push_back(*it++);

  while (state.KeepRunning()) {
    auto c = is_palindrone(list);
    benchmark::DoNotOptimize(c);
  }
}
BENCHMARK(BM_is_palindrone)->Range(BM_length_min, BM_length_max);

CTCI_MAIN();
