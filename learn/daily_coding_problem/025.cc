// This problem was asked by Facebook.
//
// Implement regular expression matching with the following special characters:
//
//   . (period) which matches any single character
//   * (asterisk) which matches zero or more of the preceding element
//
// That is, implement a function that takes in a string and a valid regular
// expression and returns whether or not the string matches the regular
// expression.
//
// For example, given the regular expression "ra." and the string "ray", your
// function should return true. The same regular expression on the string
// "raymond" should return false.
//
// Given the regular expression ".*at" and the string "chat", your function
// should return true. The same regular expression on the string "chats" should
// return false.
#include <iostream>
#include "labm8/cpp/string.h"
#include "labm8/cpp/test.h"

bool IsReMatch(const string& input, const string& re, int inputPos = 0,
               int rePos = 0) {
  if (inputPos == input.size() || rePos == re.size()) {
    // We have reached the end of either input.
    if (rePos < re.size() && re[rePos] == '*') {
      // A trailing asterisk is okay.
      return true;
    }
    return inputPos == input.size() && rePos == re.size();
  } else if (re[rePos] == '.' || re[rePos] == input[inputPos]) {
    // Character or '.' wildcard match.
    return IsReMatch(input, re, inputPos + 1, rePos + 1);
  } else if (re[rePos] == '*') {
    if (!rePos || !inputPos) {
      // Invalid regex: Starts with '*' wildcard.
      return false;
    }
    if (input[inputPos] == input[inputPos - 1]) {
      return (IsReMatch(input, re, inputPos + 1, rePos + 1) ||
              IsReMatch(input, re, inputPos + 1, rePos));
    } else {
      return IsReMatch(input, re, inputPos, rePos + 1);
    }
  }

  return false;
}

TEST(RegularExpression, EmptyString) {
  const string input = "";
  const string re = "";
  EXPECT_TRUE(IsReMatch(input, re));
}

TEST(RegularExpression, EmptyStringAndNotEmptyRegex) {
  const string input = "";
  const string re = "abc";
  EXPECT_FALSE(IsReMatch(input, re));
}

TEST(RegularExpression, StringIsMatch) {
  const string input = "abc";
  const string re = "abc";
  EXPECT_TRUE(IsReMatch(input, re));
}

TEST(RegularExpression, StringIsNotMatch) {
  const string input = "abc";
  const string re = "abd";
  EXPECT_FALSE(IsReMatch(input, re));
}

TEST(RegularExpression, StringIsSubstring) {
  const string input = "abc";
  const string re = "abcdef";
  EXPECT_FALSE(IsReMatch(input, re));
}

TEST(RegularExpression, RegexDotMatch) {
  const string input = "abc";
  const string re = "ab.";
  EXPECT_TRUE(IsReMatch(input, re));
}

TEST(RegularExpression, RegexDotMismatch) {
  const string input = "abc";
  const string re = "ab..";
  EXPECT_FALSE(IsReMatch(input, re));
}

TEST(RegularExpression, RegexAllDots) {
  const string input = "abc";
  const string re = "...";
  EXPECT_TRUE(IsReMatch(input, re));
}

TEST(RegularExpression, DotsTooShort) {
  const string input = "abc";
  const string re = "a.";
  EXPECT_FALSE(IsReMatch(input, re));
}

TEST(RegularExpression, AsteriskNoMatch) {
  const string input = "a";
  const string re = "a*";
  EXPECT_TRUE(IsReMatch(input, re));
}

TEST(RegularExpression, AsteriskMatch) {
  const string input = "aaaa";
  const string re = "a*";
  EXPECT_TRUE(IsReMatch(input, re));
}

TEST(RegularExpression, AsteriskMismatch) {
  const string input = "abc";
  const string re = "a*d";
  EXPECT_FALSE(IsReMatch(input, re));
}

TEST(RegularExpression, InvalidStartWithWildcard) {
  const string input = "abc";
  const string re = "*";
  EXPECT_FALSE(IsReMatch(input, re));
}

TEST_MAIN();
