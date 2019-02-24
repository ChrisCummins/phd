#include <iostream>
#include <cassert>
#include <string>

void inplace_reverse_string(std::string& str) {
  for (int i = 0; i < str.size() / 2; ++i) {
    int j = str.size() - 1 - i;
    std::swap(str[i], str[j]);
  }
}


std::string reverse_string(const std::string& str) {
  return std::string(str.rbegin(), str.rend());
}


int main(int argc, char **argv) {
  std::string a = "abc";
  inplace_reverse_string(a);
  assert(a == "cba");
  std::cout << a << std::endl;

  assert(reverse_string("abca") == "cba");
  std::cout << reverse_string("abc") << std::endl;

  return 0;
}
