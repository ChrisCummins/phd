/*
 * SAXPY using SYCL.
 */
#include "./trisycl.h"

#include <vector>

#include <stdlib.h>

static unsigned int seed = 0xCEC;

//
// Baseline, sequential SAXPY.
//
template <typename T>
std::vector<T> saxpy(const T& a, const std::vector<T>& x,
                     const std::vector<T>& y) {
  if (x.size() != y.size()) throw std::invalid_argument("x.size() != y.size()");

  std::vector<T> out(x.size());
  for (size_t i = 0; i < out.size(); i++) out[i] = a * x[i] + y[i];

  return out;
}

//
// SAXPY using SYCL.
//

std::vector<float> sycl_saxpy(const float& a, const std::vector<float>& x,
                              const std::vector<float>& y) {
  if (x.size() != y.size()) throw std::invalid_argument("x.size() != y.size()");

  std::vector<float> out(x.size());
  {
    cl::sycl::queue myQueue;
    cl::sycl::buffer<float> dev_a{&a, 1};
    cl::sycl::buffer<float> dev_x{x.data(), x.size()};
    cl::sycl::buffer<float> dev_y{y.data(), y.size()};
    cl::sycl::buffer<float> dev_out{out.data(), out.size()};

    myQueue.submit([&](cl::sycl::handler& cgh) {
      auto ka = dev_a.get_access<cl::sycl::access::read>(cgh);
      auto kx = dev_x.get_access<cl::sycl::access::read>(cgh);
      auto ky = dev_y.get_access<cl::sycl::access::read>(cgh);
      auto kout = dev_out.get_access<cl::sycl::access::write>(cgh);
      cgh.parallel_for(
          cl::sycl::range<1>{out.size()},
          [=](const cl::sycl::id<1> i) { kout[i] += ka[0] * kx[i] + ky[i]; });
    });
  }

  return out;
}

///////////
// Tests //
///////////

static const float test1_a = 3;
static const std::vector<float> test1_x{1, 2, 3};
static const std::vector<float> test1_y{3, 2, 1};
static const std::vector<float> test1_out{6, 8, 10};

TEST(Implementations, saxpy) {
  ASSERT_TRUE(test1_out == saxpy(test1_a, test1_x, test1_y));
}

TEST(Implementations, sycl_saxpy) {
  ASSERT_TRUE(test1_out == sycl_saxpy(test1_a, test1_x, test1_y));
}

////////////////
// Benchmarks //
////////////////

static const size_t BM_length_min = 8;
static const size_t BM_length_max = 10 << 10;

void BM_saxpy(benchmark::State& state) {
  float a = rand_r(&seed) / static_cast<float>(UINT32_MAX);
  std::vector<float> x(static_cast<size_t>(state.range(0)));
  std::vector<float> y(static_cast<size_t>(state.range(0)));

  while (state.KeepRunning()) {
    for (size_t i = 0; i < x.size(); i++) {
      x[i] = rand_r(&seed) / static_cast<float>(UINT32_MAX);
      y[i] = rand_r(&seed) / static_cast<float>(UINT32_MAX);
    }

    auto c = saxpy(a, x, y);
    benchmark::DoNotOptimize(c.data());
  }
}
BENCHMARK(BM_saxpy)->Range(BM_length_min, BM_length_max);

void BM_sycl_saxpy(benchmark::State& state) {
  float a = rand_r(&seed) / static_cast<float>(UINT32_MAX);
  std::vector<float> x(static_cast<size_t>(state.range(0)));
  std::vector<float> y(static_cast<size_t>(state.range(0)));

  while (state.KeepRunning()) {
    for (size_t i = 0; i < x.size(); i++) {
      x[i] = rand_r(&seed) / static_cast<float>(UINT32_MAX);
      y[i] = rand_r(&seed) / static_cast<float>(UINT32_MAX);
    }

    auto c = sycl_saxpy(a, x, y);
    benchmark::DoNotOptimize(c.data());
  }
}
BENCHMARK(BM_sycl_saxpy)->Range(BM_length_min, BM_length_max);

PHD_MAIN();
