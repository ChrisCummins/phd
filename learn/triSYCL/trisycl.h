#ifndef __TRISYCL_H__
#define __TRISYCL_H__

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#include <CL/sycl.hpp>
#pragma GCC diagnostic pop

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpadded"
#pragma GCC diagnostic ignored "-Wundef"
#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

#define MAIN() \
  int main(int argc, char **argv) { \
    testing::InitGoogleTest(&argc, argv); \
    const auto ret = RUN_ALL_TESTS(); \
    benchmark::Initialize(&argc, argv); \
    benchmark::RunSpecifiedBenchmarks(); \
    return ret; \
  }

#endif  // __TRISYCL_H__
