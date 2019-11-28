/*
 * Write code to remove duplicates from an unsorted linked list.
 *
 * FOLLOW UP
 *
 *   How would you solve this problem if a temporary buffer is not
 *   allowed?
 */
#include "./ctci.h"

#include <forward_list>
#include <unordered_map>

static unsigned int seed = 0xCEC;

//
// First implementation. Build a frequency table, then iterate through
// the list, deleting elements whos frequency is > 1.
//
// O(n) time, O(n) space;
//
template <typename T>
void remove_dups(std::forward_list<T>& list) {
  std::unordered_map<T, size_t> freqs;

  auto it = list.begin();
  while (it != list.end()) freqs[*it++]++;

  it = list.begin();
  auto prev = list.before_begin();

  while (it != list.end()) {
    auto next = it;
    next++;

    if (--freqs[*it] > 0) {
      list.erase_after(prev);
    } else {
      prev = it;
    }

    it = next;
  }
}

//
// In place version. For every element in the list, iterate to the end
// of the list and remove and duplicates.
//
// O(n^2) time, O(1) space.
//
template <typename T>
void inplace_remove_dups(std::forward_list<T>& list) {
  auto it = list.begin();

  while (it != list.end()) {
    auto dit = it, prev = it, next = it;
    dit++;

    while (dit != list.end()) {
      next = dit;
      next++;

      if (*dit == *it)
        list.erase_after(prev);
      else
        prev = dit;

      dit = next;
    }

    it++;
  }
}

///////////
// Tests //
///////////

TEST(Duplicates, remove_dups) {
  std::forward_list<int> l1{1, 2, 3, 4, 5, 1, 2, 6};
  const std::forward_list<int> l1a{3, 4, 5, 1, 2, 6};

  remove_dups(l1);
  ASSERT_TRUE(l1 == l1a);

  std::forward_list<int> l2{1, 1, 1, 2, 3, 4, 5, 1, 2, 6};
  const std::forward_list<int> l2a{3, 4, 5, 1, 2, 6};

  remove_dups(l2);
  ASSERT_TRUE(l2 == l2a);

  std::forward_list<int> l3{1, 1, 1};
  std::forward_list<int> l3a{1};

  remove_dups(l3);
  ASSERT_TRUE(l3 == l3a);

  std::forward_list<int> l4{1, 2, 3, 4, 5};
  const std::forward_list<int> l4a{1, 2, 3, 4, 5};

  remove_dups(l4);
  ASSERT_TRUE(l4 == l4a);
}

TEST(Duplicates, inplace_remove_dups) {
  std::forward_list<int> l1{1, 2, 3, 4, 5, 1, 2, 6};
  const std::forward_list<int> l1a{1, 2, 3, 4, 5, 6};

  inplace_remove_dups(l1);
  ASSERT_TRUE(l1 == l1a);

  std::forward_list<int> l2{1, 1, 1, 2, 3, 4, 5, 1, 2, 6};
  const std::forward_list<int> l2a{1, 2, 3, 4, 5, 6};

  inplace_remove_dups(l2);
  ASSERT_TRUE(l2 == l2a);

  std::forward_list<int> l3{1, 1, 1};
  std::forward_list<int> l3a{1};

  inplace_remove_dups(l3);
  ASSERT_TRUE(l3 == l3a);

  std::forward_list<int> l4{1, 2, 3, 4, 5};
  const std::forward_list<int> l4a{1, 2, 3, 4, 5};

  inplace_remove_dups(l4);
  ASSERT_TRUE(l4 == l4a);
}

////////////////
// Benchmarks //
////////////////

static const size_t BM_length_min = 8;
static const size_t BM_length_max = 10 << 10;

void BM_remove_dups(benchmark::State& state) {
  while (state.KeepRunning()) {
    std::forward_list<int> list;

    for (auto i = 0; i < state.range(0); i++)
      list.push_front(rand_r(&seed) % 10);

    remove_dups(list);
    benchmark::DoNotOptimize(list.front());
  }
}
BENCHMARK(BM_remove_dups)->Range(BM_length_min, BM_length_max);

void BM_inplace_remove_dups(benchmark::State& state) {
  while (state.KeepRunning()) {
    std::forward_list<int> list;

    for (auto i = 0; i < state.range(0); i++)
      list.push_front(rand_r(&seed) % 10);

    inplace_remove_dups(list);
    benchmark::DoNotOptimize(list.front());
  }
}
BENCHMARK(BM_inplace_remove_dups)->Range(BM_length_min, BM_length_max);

CTCI_MAIN();
