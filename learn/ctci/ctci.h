#ifndef CTCI_H
#define CTCI_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weverything"
#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

#include <stdlib.h>

#define CTCI_MAIN() \
  int main(int argc, char **argv) { \
    testing::InitGoogleTest(&argc, argv); \
    const auto ret = RUN_ALL_TESTS(); \
    benchmark::Initialize(&argc, argv); \
    benchmark::RunSpecifiedBenchmarks(); \
    return ret; \
  }

#endif  // CTCI_H
