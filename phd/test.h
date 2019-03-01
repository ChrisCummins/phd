// Header file for unit testing.
//
// All C++ unit tests files should include this header, which will pull in
// gtest and benchmark libraries.
#pragma once

#include "phd/app.h"

#include "benchmark/benchmark.h"
#include "gtest/gtest.h"

#ifdef TEST_MAIN
#error "TEST_MAIN already defined!"
#endif

// Inserts a main() function which runs google benchmarks and gtest suite.
#define TEST_MAIN()                                    \
  int main(int argc, char **argv) {                    \
    testing::InitGoogleTest(&argc, argv);              \
    phd::InitApp(&argc, &argv, "Test suite program."); \
    const auto ret = RUN_ALL_TESTS();                  \
    benchmark::Initialize(&argc, argv);                \
    benchmark::RunSpecifiedBenchmarks();               \
    return ret;                                        \
  }
