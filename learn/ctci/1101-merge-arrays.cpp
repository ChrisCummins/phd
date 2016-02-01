/*
 * You are given two sorted arrays, A and B, where A has a large
 * enough buffer at the end to hold B. Write a method to merge B into
 * A in sorted order.
 */
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpadded"
#pragma GCC diagnostic ignored "-Wundef"
#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

//
// First implementation. Create temporary buffer to merge into, then
// copy values over.
//
template<typename T>
void merge1(std::vector<T> &left, const size_t leftlen, std::vector<T> &right) {
  const size_t len = leftlen + right.size();
  std::vector<T> res(len);

  size_t lp = 0, rp = 0, mp = 0;
  while (lp < leftlen && rp < right.size()) {
    if (left[lp] < right[rp]) {
      res[mp] = left[lp];
      lp++;
    } else {
      res[mp] = right[rp];
      rp++;
    }
    mp++;
  }

  while (lp < leftlen)
    res[mp++] = left[lp++];
  while (rp < right.size())
    res[mp++] = right[rp++];

  for (size_t i = 0; i < len; i++)
    left[i] = res[i];
}

// Unit tests

TEST(Merge, merge1) {
  std::vector<size_t> a{0, 2, 4, 0, 0};
  std::vector<size_t> b{1, 3};
  merge1(a, 3, b);

  for (size_t i = 0; i < a.size(); i++)
    ASSERT_EQ(a[i], i);
}


// Benchmarks

static const size_t lengthMin = 8;
static const size_t lengthMax = 10 << 10;

void BM_std_inplace_merge(benchmark::State& state) {
  auto len = static_cast<size_t>(state.range_x());

  std::vector<int> a(len * 2);

  while (state.KeepRunning()) {
    int aa = 0;

    for (size_t i = 0; i < len; i++) {
      aa += arc4random() % 10;
      a[i] = aa;
    }

    for (size_t i = len; i < len * 2; i++) {
      aa += arc4random() % 10;
      a[i] = aa;
    }

    auto mid = a.begin() + static_cast<int>(len);

    std::inplace_merge(a.begin(), mid, a.end());
    benchmark::DoNotOptimize(a.data());
  }
}
BENCHMARK(BM_std_inplace_merge)->Range(lengthMin, lengthMax);

void BM_merge1(benchmark::State& state) {
  auto len = static_cast<size_t>(state.range_x());

  std::vector<int> a(len * 2);
  std::vector<int> b(len);

  while (state.KeepRunning()) {
    int aa = 0, bb = 0;
    for (size_t i = 0; i < len; i++) {
      aa += arc4random() % 10;
      bb += arc4random() % 10;

      a[i] = aa;
      b[i] = bb;
    }

    merge1(a, len, b);
    benchmark::DoNotOptimize(a.data());
  }
}
BENCHMARK(BM_merge1)->Range(lengthMin, lengthMax);


int main(int argc, char **argv) {
  // Run unit tests:
  testing::InitGoogleTest(&argc, argv);
  const auto ret = RUN_ALL_TESTS();

  // Run benchmarks:
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();

  return ret;
}
