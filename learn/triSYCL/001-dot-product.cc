/*
 * Dot product using SYCL.
 *
 * Based on triSYCL's and Ruyman Reyes' vector addition examples:
 *
 *   https://github.com/amd/triSYCL/blob/master/tests/examples/vector_add.cpp
 *   https://www.codeplay.com/portal/sycl-tutorial-1-the-vector-addition
 */
#include "./trisycl.h"

#include <iostream>
#include <vector>

#include <stdlib.h>

static unsigned int seed = 0xCEC;

//
// Baseline, sequential dot product.
//
template <typename T>
auto dot_product(const std::vector<T>& a, const std::vector<T>& b) {
  if (a.size() != b.size()) throw std::invalid_argument("a.size() != b.size()");

  float c = 0;
  for (size_t i = 0; i < a.size(); i++) c += a[i] * b[i];

  return c;
}

//
// Dot product using SYCL.
//

// float specialization
auto sycl_dot_product(const std::vector<float>& a,
                      const std::vector<float>& b) {
  if (a.size() != b.size()) throw std::invalid_argument("a.size() != b.size()");

  float c = 0;
  {  // Nested {} block encapsulates SYCL tasks
    cl::sycl::queue myQueue;

    // Create device buffers:
    cl::sycl::buffer<float> dev_a{a.data(), a.size()};
    cl::sycl::buffer<float> dev_b{b.data(), b.size()};
    cl::sycl::buffer<float> dev_c{&c, 1};

    // The command group describing all operations needed for the
    // kernel execution:
    myQueue.submit([&](cl::sycl::handler& cgh) {
      // Accessors:
      auto ka = dev_a.get_access<cl::sycl::access::read>(cgh);
      auto kb = dev_b.get_access<cl::sycl::access::read>(cgh);
      auto kc = dev_c.get_access<cl::sycl::access::read_write>(cgh);
      // Kernel:
      cgh.parallel_for(
          cl::sycl::range<1>{a.size()},
          [=](const cl::sycl::id<1> i) { kc[0] += ka[i] * kb[i]; });
    });
  }

  return c;
}

// int specialization
auto sycl_dot_product(const std::vector<int>& a, const std::vector<int>& b) {
  if (a.size() != b.size()) throw std::invalid_argument("a.size() != b.size()");

  int c = 0;
  {  // Nested {} block encapsulates SYCL tasks
    cl::sycl::queue myQueue;

    // Create device buffers:
    cl::sycl::buffer<int> dev_a{a.data(), a.size()};
    cl::sycl::buffer<int> dev_b{b.data(), b.size()};
    cl::sycl::buffer<int> dev_c{&c, 1};

    // The command group describing all operations needed for the
    // kernel execution:
    myQueue.submit([&](cl::sycl::handler& cgh) {
      // Accessors:
      auto ka = dev_a.get_access<cl::sycl::access::read>(cgh);
      auto kb = dev_b.get_access<cl::sycl::access::read>(cgh);
      auto kc = dev_c.get_access<cl::sycl::access::read_write>(cgh);
      // Kernel:
      cgh.parallel_for(
          cl::sycl::range<1>{a.size()},
          [=](const cl::sycl::id<1> i) { kc[0] += ka[i] * kb[i]; });
    });
  }

  return c;
}

///////////
// Tests //
///////////

static const std::vector<float> test1_a{.5, 2, 3, 0, -1};
static const std::vector<float> test1_b{.5, 2, 3, 1, 1};
static const float test1_c = 12.25;

static const std::vector<int> test2_a{1, 2, 3};
static const std::vector<int> test2_b{1, 2, 3};
static const int test2_c = 14;

TEST(Implementations, dot_product_float) {
  ASSERT_FLOAT_EQ(test1_c, dot_product(test1_a, test1_b));
}

TEST(Implementations, sycl_dot_product_float) {
  ASSERT_FLOAT_EQ(test1_c, sycl_dot_product(test1_a, test1_b));
}

TEST(Implementations, dot_product_int) {
  ASSERT_EQ(test2_c, dot_product(test2_a, test2_b));
}

TEST(Implementations, sycl_dot_product_int) {
  ASSERT_EQ(test2_c, sycl_dot_product(test2_a, test2_b));
}

////////////////
// Benchmarks //
////////////////

static const size_t BM_length_min = 8;
static const size_t BM_length_max = 10 << 10;

void BM_dot_product_float(benchmark::State& state) {
  std::vector<float> a(static_cast<size_t>(state.range(0)));
  std::vector<float> b(static_cast<size_t>(state.range(0)));

  while (state.KeepRunning()) {
    for (size_t i = 0; i < a.size(); i++) {
      a[i] = rand_r(&seed) / static_cast<float>(UINT32_MAX);
      b[i] = rand_r(&seed) / static_cast<float>(UINT32_MAX);
    }

    auto c = dot_product(a, b);
    benchmark::DoNotOptimize(c);
  }
}
BENCHMARK(BM_dot_product_float)->Range(BM_length_min, BM_length_max);

void BM_sycl_dot_product_float(benchmark::State& state) {
  std::vector<float> a(static_cast<size_t>(state.range(0)));
  std::vector<float> b(static_cast<size_t>(state.range(0)));

  while (state.KeepRunning()) {
    for (size_t i = 0; i < a.size(); i++) {
      a[i] = rand_r(&seed) / static_cast<float>(UINT32_MAX);
      b[i] = rand_r(&seed) / static_cast<float>(UINT32_MAX);
    }

    auto c = sycl_dot_product(a, b);
    benchmark::DoNotOptimize(c);
  }
}
BENCHMARK(BM_sycl_dot_product_float)->Range(BM_length_min, BM_length_max);

void BM_dot_product_int(benchmark::State& state) {
  std::vector<int> a(static_cast<size_t>(state.range(0)));
  std::vector<int> b(static_cast<size_t>(state.range(0)));

  while (state.KeepRunning()) {
    for (size_t i = 0; i < a.size(); i++) {
      a[i] = static_cast<int>(rand_r(&seed));
      b[i] = static_cast<int>(rand_r(&seed));
    }

    auto c = dot_product(a, b);
    benchmark::DoNotOptimize(c);
  }
}
BENCHMARK(BM_dot_product_int)->Range(BM_length_min, BM_length_max);

void BM_sycl_dot_product_int(benchmark::State& state) {
  std::vector<int> a(static_cast<size_t>(state.range(0)));
  std::vector<int> b(static_cast<size_t>(state.range(0)));

  while (state.KeepRunning()) {
    for (size_t i = 0; i < a.size(); i++) {
      a[i] = static_cast<int>(rand_r(&seed));
      b[i] = static_cast<int>(rand_r(&seed));
    }

    auto c = sycl_dot_product(a, b);
    benchmark::DoNotOptimize(c);
  }
}
BENCHMARK(BM_sycl_dot_product_int)->Range(BM_length_min, BM_length_max);

PHD_MAIN();
