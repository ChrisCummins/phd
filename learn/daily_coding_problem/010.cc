// This problem was asked by Amazon.
//
// Given an integer k and a string s, find the length of the longest substring
// that contains at most k distinct characters.
//
// For example, given s = "abcba" and k = 2, the longest substring with k
// distinct characters is "bcb".

#include <iostream>
#include <string>
#include <unordered_map>

// Time: O(n)
// Space: O(n)
int fn(const std::string& s, const int k) {
  std::unordered_map<char, int> c;
  int max = 0;
  int i = 0;

  for (int j = 0; j < s.size(); ++j) {
    // Increment the character frequency table.
    auto iter = c.find(s[j]);
    if (iter == c.end()) {
      c.insert(std::make_pair(s[j], 1));
    } else {
      ++iter->second;
    }

    // Move the left pointer along until we are within bounds.
    while (c.size() > k) {
      auto iter = c.find(s[i]);
      --iter->second;
      if (iter->second == 0) {
        c.erase(iter);
      }
      ++i;
    }

    // Update the rolling maximum.
    max = std::max(j - i + 1, max);
  }

  return max;
}

void Test(const std::string& s, const int k, const int expected) {
  auto actual = fn(s, k);
  std::cout << "fn(\"" << s << "\", " << k << ") = " << actual;
  if (actual != expected) {
    std::cout << " <- wrong! Expected: " << expected;
  }
  std::cout << std::endl;
}

int main(int argc, char** argv) {
  Test("", 0, 0);
  Test("a", 0, 0);
  Test("a", 1, 1);
  Test("a", 2, 1);
  Test("aaa", 1, 3);
  Test("abcbd", 2, 3);
  Test("abcbd", 3, 4);
  Test("abcbcbcbd", 2, 7);
  Test("aaabbcdddddddddddd", 3, 15);
  return 0;
}
