/*
 * Write code to remove duplicates from an unsorted linked list.
 *
 * FOLLOW UP
 *
 *   How would you solve this problem if a temporary buffer is not
 *   allowed?
 */
#include <forward_list>
#include <unordered_map>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpadded"
#pragma GCC diagnostic ignored "-Wundef"
#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#pragma GCC diagnostic pop


//
// First implementation. Build a frequency table, then iterate through
// the list, deleting elements whos frequency is > 1.
//
template<typename T>
void remove_dups(std::forward_list<T>& list) {
  std::unordered_map<T, size_t> freqs;

  auto it = list.begin();

  while (it != list.end())
    freqs[*it++]++;

  // Remove front elements first.
  while (freqs[list.front()] > 1) {
    freqs[list.front()]--;
    list.pop_front();
  }

  // Start iterating at the second element.
  it = list.begin();
  it++;
  auto prev = list.begin();

  while (it != list.end()) {
    auto next = it;
    next++;

    if (freqs[*it] > 1) {
      freqs[*it]--;
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
template<typename T>
void remove_dups_in_place(std::forward_list<T> &list) {
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


TEST(remove_dups, tests) {
  std::forward_list<int> l1{1, 2, 3, 4, 5, 1, 2, 6};
  const std::forward_list<int> l2{3, 4, 5, 1, 2, 6};

  remove_dups(l1);
  ASSERT_TRUE(l1 == l2);

  std::forward_list<int> l3{1, 1, 1, 2, 3, 4, 5, 1, 2, 6};
  const std::forward_list<int> l4{3, 4, 5, 1, 2, 6};

  remove_dups(l3);
  ASSERT_TRUE(l3 == l4);

  std::forward_list<int> l5{1, 1, 1};
  std::forward_list<int> l5a{1};

  remove_dups(l5);
  ASSERT_TRUE(l5 == l5a);

  std::forward_list<int> l6{1, 2, 3, 4, 5};
  const std::forward_list<int> l7{1, 2, 3, 4, 5};

  remove_dups(l6);
  ASSERT_TRUE(l6 == l7);
}


TEST(remove_dups_in_place, tests) {
  std::forward_list<int> l1{1, 2, 3, 4, 5, 1, 2, 6};
  const std::forward_list<int> l2{1, 2, 3, 4, 5, 6};

  remove_dups_in_place(l1);
  ASSERT_TRUE(l1 == l2);

  std::forward_list<int> l3{1, 1, 1, 2, 3, 4, 5, 1, 2, 6};
  const std::forward_list<int> l4{1, 2, 3, 4, 5, 6};

  remove_dups_in_place(l3);
  ASSERT_TRUE(l3 == l4);

  std::forward_list<int> l5{1, 1, 1};
  std::forward_list<int> l5a{1};

  remove_dups_in_place(l5);
  ASSERT_TRUE(l5 == l5a);

  std::forward_list<int> l6{1, 2, 3, 4, 5};
  const std::forward_list<int> l7{1, 2, 3, 4, 5};

  remove_dups_in_place(l6);
  ASSERT_TRUE(l6 == l7);
}


static const size_t lengthMin = 8;
static const size_t lengthMax = 10 << 10;

void remove_dups(benchmark::State& state) {
  while (state.KeepRunning()) {
    std::forward_list<int> list;

    for (auto i = 0; i < state.range_x(); i++)
      list.push_front(arc4random() % 10);

    remove_dups(list);
    benchmark::DoNotOptimize(list.front());
  }
}
BENCHMARK(remove_dups)->Range(lengthMin, lengthMax);


void remove_dups_in_place(benchmark::State& state) {
  while (state.KeepRunning()) {
    std::forward_list<int> list;

    for (auto i = 0; i < state.range_x(); i++)
      list.push_front(arc4random() % 10);

    remove_dups_in_place(list);
    benchmark::DoNotOptimize(list.front());
  }
}
BENCHMARK(remove_dups_in_place)->Range(lengthMin, lengthMax);


int main(int argc, char **argv) {
  // Run unit tests:
  testing::InitGoogleTest(&argc, argv);
  const auto ret = RUN_ALL_TESTS();

  // Run benchmarks:
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();

  return ret;
}
