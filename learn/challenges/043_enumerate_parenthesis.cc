// Given n pairs of parenthesis, write a function to generate all combinations
// of well-formed parenthesis.

#include "labm8/cpp/test.h"

#include <string>
#include <vector>

using std::string;
using std::vector;

// Time: O(2 ^ n)
// Space: O(n)
void EnumerateParenthesis(int n, int d, const string& s, vector<string>* res) {
  if (n) {
    EnumerateParenthesis(n - 1, d + 1, s + "(", res);
  }
  if (d) {
    EnumerateParenthesis(n, d - 1, s + ")", res);
  }

  if (!n && !d) {
    res->push_back(s);
  }
}

vector<string> EnumerateParenthesis(int n) {
  vector<string> s;
  EnumerateParenthesis(n, 0, "", &s);
  return s;
}

TEST(EnumerateParenthesis, Zero) {
  EXPECT_EQ(EnumerateParenthesis(0), vector<string>({""}));
}

TEST(EnumerateParenthesis, One) {
  EXPECT_EQ(EnumerateParenthesis(1), vector<string>({"()"}));
}

TEST(EnumerateParenthesis, Two) {
  EXPECT_EQ(EnumerateParenthesis(2), vector<string>({"(())", "()()"}));
}

TEST(EnumerateParenthesis, Three) {
  EXPECT_EQ(EnumerateParenthesis(3), vector<string>({
                                         "((()))",
                                         "(()())",
                                         "(())()",
                                         "()(())",
                                         "()()()",
                                     }));
}

TEST_MAIN();
