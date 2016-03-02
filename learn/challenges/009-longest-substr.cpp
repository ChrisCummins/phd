/*
 * Given a string s and a number N, find the longest substring of s
 * with maximum N unique characters.
 */
#include <algorithm>
#include <iostream>
#include <string>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpadded"
#pragma GCC diagnostic ignored "-Wundef"
#include <gtest/gtest.h>
#pragma GCC diagnostic pop


// Can't remember how to get the number of unique values in
// std::string::char_type:
static const size_t nchars = 255;


std::string longest_substr(const std::string& s, const size_t n) {
  size_t start = 0, maxlen = 1, i = 0, j = 0, len = 0;

  while (i < s.length() - maxlen) {
    std::bitset<nchars> uniq{0};

    for (j = i; j < s.length(); j++) {
      uniq[static_cast<size_t>(s[j])] = true;
      if (uniq.count() > n)
        break;
    }

    len = j - i;

    if (len > maxlen) {
      start = i;
      maxlen = len;
    }

    i++;
  }

  return s.substr(start, maxlen);
}


TEST(basic, longest_substr) {
  ASSERT_EQ("ddddd", longest_substr("abcddddd", 1));
  ASSERT_EQ("cddddd", longest_substr("abcddddd", 2));
  ASSERT_EQ("cccddddd", longest_substr("abcccddddd", 2));
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
