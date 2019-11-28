#include <array>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>

static const std::string testZips[] = {std::string("TX 34254"),
                                       std::string("not a zip"),
                                       std::string("  TX12345"),
                                       std::string("DN 23456-2345 "),
                                       std::string("AB 98345-2342"),
                                       std::string("foo bar"),
                                       std::string("")};

int main(int argc, char **argv) {
  std::string s = "Hello, world!";
  std::basic_string<char> a =
      "std::string is "
      "an alias to std::basic_string<char>";

  // String printing.
  printf("C-style: %s\n", s.c_str());
  std::cout << "Streams: " << s << std::endl;
  std::cout << std::endl << a << std::endl;

  // Regex matching.
  std::cout << std::endl << "Valid ZIP codes:" << std::endl;

  std::string b = (R"(^\s*(\w{2}\s*\d{5}(-\d{4})?)\s*$)");
  const std::regex pat(b);

  int lineno = 0;
  std::smatch matches;
  for (auto &str : testZips) {
    lineno++;
    if (regex_match(str, matches, pat))
      std::cout << lineno << ": " << matches[1] << std::endl;
    else
      std::cout << lineno << ": " << std::endl;
  }

  // String stream.
  std::ostringstream oss;
  oss << '\n'
      << "this "
      << "is"
      << " a " << 's' << "tring stream!!!" << 1;
  std::cout << oss.str() << std::endl;

  return 0;
}
