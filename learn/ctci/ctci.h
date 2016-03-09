#ifndef CTCI_H
#define CTCI_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated"
#pragma GCC diagnostic ignored "-Wfloat-equal"
#pragma GCC diagnostic ignored "-Wmissing-noreturn"
#pragma GCC diagnostic ignored "-Wpadded"
#pragma GCC diagnostic ignored "-Wshift-sign-overflow"
#pragma GCC diagnostic ignored "-Wundef"
#pragma GCC diagnostic ignored "-Wused-but-marked-unused"
#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

#define CTCI_MAIN() \
  int main(int argc, char **argv) { \
    testing::InitGoogleTest(&argc, argv); \
    const auto ret = RUN_ALL_TESTS(); \
    benchmark::Initialize(&argc, argv); \
    benchmark::RunSpecifiedBenchmarks(); \
    return ret; \
  }

#endif  // CTCI_H
