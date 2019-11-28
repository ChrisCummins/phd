/*
 * You are given two sorted arrays, A and B, where A has a large
 * enough buffer at the end to hold B. Write a method to merge B into
 * A in sorted order.
 */
#include "./ctci.h"

#include <iterator>
#include <vector>

static unsigned int seed = 0xCEC;

//
// First implementation. Create temporary buffer to merge into, then
// copy values over.
//
// O(n) time, O(n) space.
//
template <typename T>
void inplace_merge(std::vector<T>& left, const size_t leftlen,
                   std::vector<T>& right) {
  std::vector<T> res(leftlen + right.size());
  const auto _leftlen =
      static_cast<typename std::vector<T>::iterator::difference_type>(leftlen);

  auto lit = left.begin(), rit = right.begin(), oit = res.begin();
  while (lit != left.begin() + _leftlen && rit < right.end())
    if (*lit < *rit)
      *oit++ = *lit++;
    else
      *oit++ = *rit++;

  while (lit != left.begin() + _leftlen) *oit++ = *lit++;
  while (rit != right.end()) *oit++ = *rit++;

  lit = left.begin();
  oit = res.begin();
  while (lit != left.end()) *lit++ = *oit++;
}

///////////
// Tests //
///////////

TEST(Merge, merge) {
  std::vector<size_t> a{0, 2, 4, 0, 0};
  std::vector<size_t> b{1, 3};
  inplace_merge(a, 3, b);

  for (size_t i = 0; i < a.size(); i++) ASSERT_EQ(a[i], i);
}

////////////////
// Benchmarks //
////////////////

static const size_t BM_length_min = 8;
static const size_t BM_length_max = 10 << 10;

void BM_std_inplace_merge(benchmark::State& state) {
  auto len = static_cast<size_t>(state.range(0));
  std::vector<int> a(len * 2);

  while (state.KeepRunning()) {
    int aa = 0;

    for (size_t i = 0; i < 2 * len; i++) {
      aa += rand_r(&seed) % 10;
      a[i] = aa;
    }

    auto mid = a.begin() + static_cast<int>(len);
    std::inplace_merge(a.begin(), mid, a.end());
    benchmark::DoNotOptimize(a.data());
  }
}
BENCHMARK(BM_std_inplace_merge)->Range(BM_length_min, BM_length_max);

void BM_std_merge(benchmark::State& state) {
  auto len = static_cast<size_t>(state.range(0));
  std::vector<int> a(len);
  std::vector<int> b(len);
  std::vector<int> out(2 * len);

  while (state.KeepRunning()) {
    int aa = 0, bb = 0;
    for (size_t i = 0; i < len; i++) {
      aa += rand_r(&seed) % 10;
      bb += rand_r(&seed) % 10;
      a[i] = aa;
      b[i] = bb;
    }

    std::merge(a.begin(), a.end(), b.begin(), b.end(), out.begin());
    benchmark::DoNotOptimize(out.data());
  }
}
BENCHMARK(BM_std_merge)->Range(BM_length_min, BM_length_max);

void BM_merge(benchmark::State& state) {
  auto len = static_cast<size_t>(state.range(0));
  std::vector<int> a(len * 2);
  std::vector<int> b(len);

  while (state.KeepRunning()) {
    int aa = 0, bb = 0;
    for (size_t i = 0; i < len; i++) {
      aa += rand_r(&seed) % 10;
      bb += rand_r(&seed) % 10;

      a[i] = aa;
      b[i] = bb;
    }

    inplace_merge(a, len, b);
    benchmark::DoNotOptimize(a.data());
  }
}
BENCHMARK(BM_merge)->Range(BM_length_min, BM_length_max);

CTCI_MAIN();
