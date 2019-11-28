/*
 * Assume you have a method isSubstring which checks if one word is a
 * substring of another. Given two strings, s1 and s2, write code to
 * check if s2 is a rotation of s1 using only one call to isSubstring
 * (e.g. "waterbottle" is a rotation of "erbottlewat").
 */
#include "./ctci.h"

#include <string>

//
// Return whether a is a substring of b.
//
bool isSubstring(const std::string a, const std::string b) {
  return b.find(a) != std::string::npos;
}

//
// Return whether a is a rotation of b.
//
// O(n) time, O(n) space.
//
bool isRotation(const std::string& a, const std::string& b) {
  // Strings must be the same size.
  if (a.size() != b.size()) return false;

  return isSubstring(a, b + b);
}

///////////
// Tests //
///////////

TEST(Rotation, isSubstring) {
  ASSERT_TRUE(isSubstring("abc", "abc"));
  ASSERT_TRUE(isSubstring("ab", "abc"));
  ASSERT_TRUE(isSubstring("b", "abc"));

  ASSERT_FALSE(isSubstring("cba", "abc"));
  ASSERT_FALSE(isSubstring("cb", "abc"));
  ASSERT_FALSE(isSubstring("d", "abc"));
}

TEST(Rotation, isRotation) {
  ASSERT_TRUE(isRotation("erbottlewat", "waterbottle"));
  ASSERT_TRUE(isRotation("mat the cat sat on the ", "the cat sat on the mat "));
  ASSERT_TRUE(isRotation("a", "a"));

  ASSERT_FALSE(isRotation("foo", "bar"));
  ASSERT_FALSE(isRotation("car", "carr"));
  ASSERT_FALSE(isRotation("d", "abc"));
}

CTCI_MAIN();
