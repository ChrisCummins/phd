// -*-c++-*-
//
// Common test header.
//
#ifndef __USTL_TEST_H__
#define __USTL_TEST_H__

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpadded"
#pragma GCC diagnostic ignored "-Wundef"
#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

//
// Helper functions & objects.
//

template<typename T>
class InverseComp {
 public:
  bool operator()(const T &a, const T &b) { return a > b; }
};

template<typename T>
class Comparable {
 public:
  Comparable() : data() {}
  explicit Comparable(const T &_data) : Comparable(_data, 0) {}
  Comparable(const T &_data, const T &_nc) : data(_data), nc(_nc) {}
  ~Comparable() {}

  bool operator<(const Comparable &rhs) const { return data < rhs.data; }
  bool operator==(const Comparable &rhs) const { return data == rhs.data; }

  T data;
  T nc;
};

bool inverse_comp(const int &a, const int &b);

// lambdas:
const auto is_even = [](const int &x) -> bool{ return !(x % 2); };
const auto increment = [](int &x) -> void{ x++; };

template<typename VectorType, typename T>
bool vector_equal(const VectorType &v, std::initializer_list<T> il) {
  auto vit = v.begin();
  auto ilit = il.begin();

  while (vit != v.end())
    if (*vit++ != *ilit++)
      return false;
  return true;
}

#endif  // __USTL_TEST_H__
