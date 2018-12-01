// Header file for unit testing.
//
// All C++ unit tests files should include this header, which will pull in
// gtest and benchmark libraries.
#pragma once

#include "benchmark/benchmark.h"
#include "gtest/gtest.h"

#ifdef TEST_MAIN
#error "TEST_MAIN already defined!"
#endif

// Inserts a main() function which runs google benchmarks and gtest suite.
#define TEST_MAIN() \
  int main(int argc, char **argv) { \
    testing::InitGoogleTest(&argc, argv); \
    const auto ret = RUN_ALL_TESTS(); \
    benchmark::Initialize(&argc, argv); \
    benchmark::RunSpecifiedBenchmarks(); \
    return ret; \
  }

namespace test {
namespace debug {

// Debug type:
template<typename T> struct debug_t {};

}  // debug namespace
}  // test namespace

// Macros for debugging types:
//
// Fatally crash the compiler by attempting to construct an object of
// 'type' using an unknown argument.
#define PRINT_TYPENAME(type) type _____{test::debug::debug_t<type>};
//
// Fatally crash the compiler by attempting to cast 'variable' to an
// unknown type.
#define PRINT_TYPE(variable) static_cast<test::debug::debug_t<type>>(variable);
