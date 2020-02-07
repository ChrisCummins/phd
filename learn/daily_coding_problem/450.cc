// This problem was asked by Google.
//
// You're given a string consisting solely of (, ), and *. * can represent
// either a (, ), or an empty string. Determine whether the parentheses are
// balanced.
//
// For example, (()* and (*) are balanced. )*( is not balanced.
#include <string>

#include "labm8/cpp/logging.h"
#include "labm8/cpp/test.h"

using std::string;

// Recursive approach, Fibonacci style.
//
// Time: O(2 ^ n)
// Space: O(n)
bool IsBalanced(const string& str, int i, int count) {
  if (str.size() == i) {
    return count == 0;
  }

  if (str[i] == '(') {
    return IsBalanced(str, i + 1, count + 1);
  } else if (str[i] == ')') {
    return IsBalanced(str, i + 1, count - 1);
  } else if (str[i] == '*') {
    return IsBalanced(str, i + 1, count + 1) ||
           IsBalanced(str, i + 1, count - 1);
  } else {  // The question didn't ask for this, but let's allow non {()*}
            // characters.
    return IsBalanced(str, i + 1, count);
  }
}

bool IsBalanced(const string& str) { return IsBalanced(str, 0, 0); }

TEST(BalancedBrackets, EmptyString) { EXPECT_TRUE(IsBalanced("")); }

TEST(BalancedBrackets, StringsWithoutBrackets) {
  EXPECT_TRUE(IsBalanced("a"));
  EXPECT_TRUE(IsBalanced("ab"));
  EXPECT_TRUE(IsBalanced("abc"));
  EXPECT_TRUE(IsBalanced("abd"));
}

TEST(BalancedBrackets, SingleBracketStrings) {
  EXPECT_FALSE(IsBalanced("("));
  EXPECT_FALSE(IsBalanced(")"));
  EXPECT_FALSE(IsBalanced("*"));
}

TEST(BalancedBrackets, MulticharUnbalanced) {
  EXPECT_FALSE(IsBalanced("(("));
  EXPECT_FALSE(IsBalanced("(()"));
  EXPECT_FALSE(IsBalanced("((*"));
  EXPECT_FALSE(IsBalanced("*))"));
  EXPECT_FALSE(IsBalanced("))"));
}

TEST(BalancedBrackets, MulticharBalanced) {
  EXPECT_TRUE(IsBalanced("()"));
  EXPECT_TRUE(IsBalanced("(())"));
  EXPECT_TRUE(IsBalanced("(*"));
  EXPECT_TRUE(IsBalanced("*)"));
  EXPECT_TRUE(IsBalanced("(*))"));
  EXPECT_TRUE(IsBalanced("((**"));
  EXPECT_TRUE(IsBalanced("**"));
  EXPECT_TRUE(IsBalanced("****"));
}

TEST(BalancedBrackets, MulticharBalancedWithIgnored) {
  EXPECT_TRUE(IsBalanced(" ( ) "));
  EXPECT_TRUE(IsBalanced("Hello (World)"));
  EXPECT_TRUE(IsBalanced("Is this really balanced? (*"));
  EXPECT_TRUE(IsBalanced("* this one starts with an asterisk!)"));
}

TEST_MAIN();
