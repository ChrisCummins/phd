/*
 * Given a real number between 0 and 1 (e.g. 0.72) that is passed in
 * as a double, print the binary representation. If the number cannot
 * be represented accurately in binary with at most 32 characters,
 * print "ERROR".
 */
#include "./ctci.h"

#include <iostream>
#include <set>
#include <string>

/*
 * TODO: This is not a solution for the actual problem! What I have
 * done here is implement a function which prints the binary _data_
 * from a double.
 */
template <typename T>
std::ostream &print_bits(std::ostream &out, const T &d,
                         std::initializer_list<size_t> il = {}) {
  const std::size_t size = sizeof(T);
  const char *const data = reinterpret_cast<const char *>(&d);
  const size_t nbits = size * 8;
  std::set<size_t> spaces{il.begin(), il.end()};

  for (int j = size - 1; j >= 0; j--) {
    for (int i = 7; i >= 0; i--) {
      out << ((data[j] & (1 << i)) ? '1' : '0');

      // Print spaces where appropriate.
      const size_t k = static_cast<size_t>(j) * 8 + static_cast<size_t>(i);
      if (spaces.find(nbits - k) != spaces.end()) out << ' ';
    }
  }

  return out;
}

//
// Template specialisation to add separator spaces between sign bit,
// exponent, and fraction.
//
std::ostream &print_bits(std::ostream &out, const double &d) {
  return print_bits(out, d, {1, 11});
}

//
// Template specialisation to add separator spaces between bytes.
//
std::ostream &print_bits(std::ostream &out, const int &d) {
  return print_bits(out, d, {8, 16, 24});
}

///////////
// Tests //
///////////

TEST(challenge, tests) {
  std::ostringstream stream;

  std::string out =
      "1 0111111111 10000000000000000000000000000000000000000000000000000";
  print_bits(stream, -1.0);
  ASSERT_EQ(stream.str(), out);
  stream.str(std::string());  // clear stream

  out = "0 0111111111 10000000000000000000000000000000000000000000000000000";
  print_bits(stream, 1.0);
  ASSERT_EQ(stream.str(), out);
  stream.str(std::string());  // clear stream

  out = "0 1000000000 00000000000000000000000000000000000000000000000000000";
  print_bits(stream, 2.0);
  ASSERT_EQ(stream.str(), out);
  stream.str(std::string());  // clear stream

  out = "0 1000000110 10011100010000000000000000000000000000000000000000000";
  print_bits(stream, 20000.0);
  ASSERT_EQ(stream.str(), out);
  stream.str(std::string());  // clear stream

  out = "1 1000000110 10011100010000000000000000000000000000000000000000000";
  print_bits(stream, -20000.0);
  ASSERT_EQ(stream.str(), out);
  stream.str(std::string());  // clear stream

  // char test

  out = "01100011";
  print_bits(stream, 'c');
  ASSERT_EQ(stream.str(), out);
  stream.str(std::string());  // clear stream

  // int test

  out = "00000000 00000000 00000000 00000101";
  print_bits(stream, 5);
  ASSERT_EQ(stream.str(), out);
  stream.str(std::string());  // clear stream
}

CTCI_MAIN();
