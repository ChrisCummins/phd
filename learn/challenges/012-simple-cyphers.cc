/*
 * Write a program which accepts cyphertext from a Caesar shifted
 * plaintext input, and returns the most likely decrypted plaintext.
 */
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>

#include "labm8/cpp/test.h"

// Data types:
using char_freqs = std::array<double, 26>;
using perms = std::array<std::string, 26>;

std::ostream& operator<<(std::ostream& out, const char_freqs& freqs) {
  for (const auto& f : freqs) std::cout << f << ' ';
  std::cout << std::endl;
  return out;
}

std::ostream& operator<<(std::ostream& out, const perms& p) {
  for (const auto& str : p) std::cout << str << std::endl;

  return out;
}

// English letter frequencies,
// source: https://en.wikipedia.org/wiki/Letter_frequency
static const char_freqs english_char_freqs{
    0.08167,  // a
    0.01492,  // b
    0.02782,  // c
    0.04253,  // d
    0.12702,  // e
    0.02228,  // f
    0.02015,  // g
    0.06094,  // h
    0.06966,  // i
    0.00153,  // j
    0.00772,  // k
    0.04025,  // l
    0.02406,  // m
    0.06749,  // n
    0.07507,  // o
    0.01929,  // p
    0.00095,  // q
    0.05987,  // r
    0.06327,  // s
    0.09056,  // t
    0.02758,  // u
    0.00978,  // v
    0.02361,  // w
    0.00150,  // x
    0.01974,  // y
    0.00074   // z
};

// Rotate plaintext by 'shift' characters.
std::string caesar_shift(const std::string& plaintext, const int shift) {
  std::string text = plaintext;

  for (auto& c : text) {
    if (c >= 'A' && c <= 'Z')
      c = 'A' + ((c - 'A' + shift) % 26);
    else if (c >= 'a' && c <= 'z')
      c = 'a' + ((c - 'a' + shift) % 26);
  }

  return text;
}

// Compute distribution of characters
char_freqs char_distribution(const std::string& text) {
  char_freqs freqs{0};

  for (const auto& c : text) {
    unsigned int index;
    if (c >= 'A' && c <= 'Z')
      index = static_cast<unsigned int>(c - 'A');
    else if (c >= 'a' && c <= 'z')
      index = static_cast<unsigned int>(c - 'a');
    else
      continue;

    freqs[index] += 1;
  }

  // Get the number of characters (note that char count != strength)
  const int nchars = std::accumulate(std::begin(freqs), std::end(freqs), 0);

  // Normalise frequencies:
  for (auto& f : freqs) f /= nchars;

  return freqs;
}

// Generate all permutations of a cyphertext.
perms permutations(const std::string& cyphertext) {
  perms p;

  p[0] = cyphertext;
  for (size_t i = 1; i < 26; i++)
    p[i] = caesar_shift(cyphertext, static_cast<int>(i));

  return p;
}

// Computer difference between two distributions in range [0,1] as sum
// of differences between corresponding values.
double linear_diff(const char_freqs& a, const char_freqs& b) {
  double diff{0};

  for (size_t i = 0; i < 26; i++) diff += std::abs(a[i] - b[i]);

  return diff / 2;
}

std::string crack(const std::string& cyphertext,
                  std::function<double(const char_freqs&, const char_freqs&)>
                      diff_fn = linear_diff,
                  const char_freqs& distribution = english_char_freqs) {
  double mindiff = DBL_MAX;
  std::string plaintext;

  for (int i = 0; i < 26; i++) {
    std::string text = caesar_shift(cyphertext, i);
    const double diff = diff_fn(char_distribution(text), distribution);
    if (diff < mindiff) {
      mindiff = diff;
      plaintext = text;
    }
  }

  return plaintext;
}

///////////
// Tests //
///////////

TEST(simple_cyphers, ceaser_shift) {
  ASSERT_EQ("def", caesar_shift("abc", 3));
  ASSERT_EQ("abc", caesar_shift("abc", 26));
  // FIXME: ASSERT_EQ("zab", caesar_shift("abc", -1));
  ASSERT_EQ("GUR PNG fng ba GUR ZNG.",
            caesar_shift("THE CAT sat on THE MAT.", 13));
}

TEST(simple_cyphers, char_distribution) {
  const auto f1 = char_distribution("abc");
  const auto f2 = char_distribution(" __abc!!.   ");

  ASSERT_DOUBLE_EQ(f1[0], 0.33333333333333331);
  ASSERT_DOUBLE_EQ(f1[0], 0.33333333333333331);
  ASSERT_DOUBLE_EQ(f1[0], 0.33333333333333331);
  ASSERT_TRUE(f1 == f2);
}

TEST(simple_cyphers, linear_diff) {
  ASSERT_DOUBLE_EQ(
      0, linear_diff(char_distribution("abc"), char_distribution("abc")));
  ASSERT_DOUBLE_EQ(
      1, linear_diff(char_distribution("abc"), char_distribution("def")));
  ASSERT_DOUBLE_EQ(0.66666666666666663, linear_diff(char_distribution("abc"),
                                                    char_distribution("cde")));
}

static const std::string sonnet{
    R"(Let me not to the marriage of true minds Admit impediments.
     Love is not love Which alters when it alteration finds,
     Or bends with the remover to remove: O no; it is an ever-fixed mark,
     That looks on tempests, and is never shaken;
     It is the star to every wandering bark,
     Whose worth's unknown, although his height be taken.
     Love's not Time's fool, though rosy lips and cheeks
     Within his bending sickle's compass come;
     Love alters not with his brief hours and weeks,
     But bears it out even to the edge of doom.
     If this be error and upon me proved,
     I never writ, nor no man ever loved.)"};

TEST(simple_cyphers, permutations) {
  ASSERT_EQ("defend the east wall of the castle",
            crack("vwxwfv lzw wskl osdd gx lzw uskldw"));
  ASSERT_EQ(sonnet, crack(caesar_shift(sonnet, 13)));
}

TEST_MAIN();
