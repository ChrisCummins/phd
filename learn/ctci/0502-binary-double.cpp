/*
 * Given a real number between 0 and 1 (e.g. 0.72) that is passed in
 * as a double, print the binary representation. If the number cannot
 * be represented accurately in binary with at most 32 characters,
 * print "ERROR".
 */
#include <iostream>
#include <string>
#include <set>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpadded"
#pragma GCC diagnostic ignored "-Wundef"
#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

/*
 * TODO: This is not a solution for the actual problem! What I have
 * done here is implement a function which prints the binary _data_
 * from a double.
 */
template<typename T>
std::ostream& print_bits(std::ostream &out, const T &d,
                         std::set<size_t> spaces = std::set<size_t>()) {
  const std::size_t size = sizeof(T);
  const char *const data = reinterpret_cast<const char *>(&d);
  const size_t nbits = size * 8;

  for (int j = size - 1; j >= 0; j--) {
    for (int i = 7; i >= 0; i--) {
      out << ((data[j] & (1 << i)) ? '1' : '0');

      // Print spaces where appropriate.
      const size_t k = static_cast<size_t>(j) * 8 + static_cast<size_t>(i);
      if (spaces.find(nbits - k) != spaces.end())
        out << ' ';
    }
  }

  return out;
}


// Template specialisation to add separator spaces between sign bit,
// exponent, and fraction.
std::ostream& print_bits(std::ostream &out, const double &d) {
  return print_bits(out, d, std::set<size_t>({1, 11}));
}

// Template specialisation to add separator spaces between bytes.
std::ostream& print_bits(std::ostream &out, const int &d) {
  return print_bits(out, d, std::set<size_t>({8, 16, 24}));
}


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


int main(int argc, char **argv) {
  // Run unit tests:
  testing::InitGoogleTest(&argc, argv);
  const auto ret = RUN_ALL_TESTS();

  // Run benchmarks:
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();

  return ret;
}
