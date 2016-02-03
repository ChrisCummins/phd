/*
 * Assume you have a method isSubstring which checks if one word is a
 * substring of another. Given two strings, s1 and s2, write code to
 * check if s2 is a rotation of s1 using only one call to isSubstring
 * (e.g. "waterbottle" is a rotation of "erbottlewat").
 */
#include <algorithm>
#include <string>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpadded"
#pragma GCC diagnostic ignored "-Wundef"
#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

//
// Return whether a is a substring of b.
//
bool isSubstring(const std::string a, const std::string b) {
  return b.find(a) != std::string::npos;
}

//
// Return whether a is a rotation of b.
//
bool isRotation(const std::string a, const std::string b) {
  // Strings must be the same size.
  if (a.size() != b.size())
    return false;

  return isSubstring(a, b + b);
}

TEST(isSubstring, tests) {
  ASSERT_TRUE(isSubstring("abc", "abc"));
  ASSERT_TRUE(isSubstring("ab", "abc"));
  ASSERT_TRUE(isSubstring("b", "abc"));

  ASSERT_FALSE(isSubstring("cba", "abc"));
  ASSERT_FALSE(isSubstring("cb", "abc"));
  ASSERT_FALSE(isSubstring("d", "abc"));
}

TEST(isRotation, tests) {
  ASSERT_TRUE(isRotation("erbottlewat", "waterbottle"));
  ASSERT_TRUE(isRotation("mat the cat sat on the ", "the cat sat on the mat "));
  ASSERT_TRUE(isRotation("a", "a"));

  ASSERT_FALSE(isRotation("foo", "bar"));
  ASSERT_FALSE(isRotation("car", "carr"));
  ASSERT_FALSE(isRotation("d", "abc"));
}

int main(int argc, char **argv) {
  // Run unit tests:
  testing::InitGoogleTest(&argc, argv);
  const auto ret = RUN_ALL_TESTS();

  // Run benchmarks:
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();

  return ret;
}
