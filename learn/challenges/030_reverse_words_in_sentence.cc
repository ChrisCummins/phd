// Reverse words in sentence without using library.
//
// Example: "The cat sat on the matt."
// to:      "matt the on sat cat The"
#include <assert.h>

#include <iostream>
#include <string>

using std::string;

// Time: O(n) where n is the size of input string.
// Space: O(n) where n is the size of the string.
string ReverseWordsInSentence(const string& s) {
  assert(s.size());

  string out;
  out.reserve(s.size());

  // Iterate backwards through sentence. Find word boundary and append it.
  int j = s.size() - 1;
  for (int i = j; i >= 0; --i) {
    if (s[i] == ' ') {
      int len = j - i;
      out += s.substr(i + 1, len) + " ";
      j -= len + 1;
    }
  }
  // Append last word.
  out += s.substr(0, j + 1) + " ";

  return out;
}

int main() {
  std::cout << ReverseWordsInSentence("The cat sat on the matt") << std::endl;
  std::cout << ReverseWordsInSentence("The") << std::endl;
  std::cout << ReverseWordsInSentence(" ") << std::endl;
}
