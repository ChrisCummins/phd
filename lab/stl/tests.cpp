#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpadded"
#pragma GCC diagnostic ignored "-Wundef"
#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

#include <array>

#include <ustl/array>

// Array tests

TEST(array, size) {
  ustl::array<int, 10> a;
  ustl::array<char, 1> b;
  ustl::array<int, 100> c;
  ustl::array<int, 0> d;

  ASSERT_EQ(10u, a.size());
  ASSERT_EQ(1u, b.size());
  ASSERT_EQ(100u, c.size());
  ASSERT_EQ(0u, d.size());
}

TEST(array, max_size) {
  ustl::array<int, 10> a;
  ustl::array<char, 1> b;
  ustl::array<int, 100> c;
  ustl::array<int, 0> d;

  ASSERT_EQ(10u, a.max_size());
  ASSERT_EQ(a.size(), a.max_size());
  ASSERT_EQ(1u, b.max_size());
  ASSERT_EQ(b.size(), b.max_size());
  ASSERT_EQ(100u, c.max_size());
  ASSERT_EQ(c.size(), c.max_size());
  ASSERT_EQ(0u, d.max_size());
  ASSERT_EQ(d.size(), d.max_size());
}

TEST(array, empty) {
  ustl::array<int, 10> a;
  ustl::array<char, 1> b;
  ustl::array<int, 100> c;
  ustl::array<int, 0> d;

  ASSERT_EQ(false, a.empty());
  ASSERT_EQ(false, b.empty());
  ASSERT_EQ(false, c.empty());
  ASSERT_EQ(true,  d.empty());
}

TEST(array, subscript) {
  ustl::array<int, 2> a;

  // Set values:
  a[0] = 1;
  a[1] = 2;

  // Test values:
  ASSERT_EQ(1, a[0]);
  ASSERT_EQ(2, a[1]);
}

TEST(array, at) {
  ustl::array<int, 2> a;
  ustl::array<int, 0> b;

  // Set values:
  a[0] = 1;
  a[1] = 2;

  // Test values:
  ASSERT_EQ(1, a.at(0));
  ASSERT_EQ(2, a.at(1));

  // Out of range tests:
  try {
    a.at(2) = 3;
    FAIL();
  } catch (std::out_of_range) {}

  try {
    b.at(0) = 1;
    FAIL();
  } catch (std::out_of_range) {}
}

TEST(array, front) {
  ustl::array<int, 3> a;

  // Set values:
  a[0] = 1;
  a[1] = 2;
  a[2] = 3;

  // Test values:
  ASSERT_EQ(1, a.front());
}

TEST(array, back) {
  ustl::array<int, 3> a;

  // Set values:
  a[0] = 1;
  a[1] = 2;
  a[2] = 3;

  // Test values:
  ASSERT_EQ(3, a.back());
}

TEST(array, data) {
  ustl::array<int, 3> a;

  // Set values:
  a[0] = 1;
  a[1] = 2;
  a[2] = 3;

  // Get pointer:
  auto *d = a.data();

  // Test values:
  ASSERT_EQ(1, d[0]);
  ASSERT_EQ(2, d[1]);
  ASSERT_EQ(3, d[2]);
}

// Benchmarks

void BM_baseline(benchmark::State& state) {
}
BENCHMARK(BM_baseline);

int main(int argc, char **argv) {
    // Run unit tests:
    testing::InitGoogleTest(&argc, argv);
    const auto ret = RUN_ALL_TESTS();

    // Run benchmarks:
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();

    return ret;
}
