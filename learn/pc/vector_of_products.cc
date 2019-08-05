#include "labm8/cpp/test.h"

#include <vector>

// Given an array of integers, return a new array such that each element at
// index i of the new array is the product of all the numbers in the original
// array except the one at i.
//
// For example, if our input was [1, 2, 3, 4, 5], the expected output would be
// [120, 60, 40, 30, 24]. If our input was [3, 2, 1], the expected output would
// be [2, 3, 6].
std::vector<int> VectorOfProducts(const std::vector<int>& vec) {
  std::vector<int> out(vec.size());

  int product = 1;
  for (auto i : vec) {
    product *= i;
  }

  for (int i = 0; i < out.size(); ++i) {
    out[i] = product / vec[i];
  }

  return out;
}

TEST(VectorOfProducts, EmptyListReturnsEmptyList) {
  std::vector<int> v;
  EXPECT_EQ(VectorOfProducts(v), std::vector<int>({}));
}

TEST(VectorOfProducts, SingleElementListReturnsTheSame) {
  std::vector<int> v{1};
  EXPECT_EQ(VectorOfProducts(v), std::vector<int>({1}));
}

TEST(VectorOfProducts, OneTwoThreeList) {
  std::vector<int> v{1, 2, 3};
  EXPECT_EQ(VectorOfProducts(v), std::vector<int>({6, 3, 2}));
}

// Follow-up: what if you can't use division?

std::vector<int> VectorOfProductsWithoutDivision(const std::vector<int>& vec) {
  std::vector<int> prefixes(vec.size());
  std::vector<int> suffixes(vec.size());

  // Create a vector of prefixes products, where element i is the product of
  // all elements up to and including i:
  //   a[i] * a[i-1] * a[i-2] * ... * a[0].
  int product = 1;
  for (int i = 0; i < prefixes.size(); ++i) {
    product *= vec[i];
    prefixes[i] = product;
  }

  // Create a vector of suffix products, where element i is the product of all
  // elements from i to the end of the list:
  //   a[i] * a[i+1] * a[i+2] * ... * a[n].
  product = 1;
  for (int i = suffixes.size() - 1; i >= 0; --i) {
    product *= vec[i];
    suffixes[i] = product;
  }

  std::vector<int> out(vec.size());
  for (int i = 0; i < vec.size(); ++i) {
    int prefix = i ? prefixes[i - 1] : 1;  // bounds checked access.
    int suffix = i < vec.size() - 1 ? suffixes[i + 1] : 1;
    out[i] = prefix * suffix;
  }

  return out;
}

TEST(VectorOfProductsWithoutDivision, EmptyListReturnsEmptyList) {
  std::vector<int> v;
  EXPECT_EQ(VectorOfProductsWithoutDivision(v), std::vector<int>({}));
}

TEST(VectorOfProductsWithoutDivision, SingleElementListReturnsTheSame) {
  std::vector<int> v{1};
  EXPECT_EQ(VectorOfProductsWithoutDivision(v), std::vector<int>({1}));
}

TEST(VectorOfProductsWithoutDivision, OneTwoThreeList) {
  std::vector<int> v{1, 2, 3};
  EXPECT_EQ(VectorOfProductsWithoutDivision(v), std::vector<int>({6, 3, 2}));
}

TEST_MAIN();
