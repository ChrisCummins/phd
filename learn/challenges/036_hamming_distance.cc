// Write a function to compute the hamming distance of two vectors.
#include <vector>

#include "labm8/cpp/logging.h"
#include "labm8/cpp/test.h"

using std::vector;

template <typename T>
int HammingDistance(const vector<T>& a, const vector<T>& b) {
  if (a.size() != b.size()) {
    return -1;
  }

  int diff = 0;

  size_t nbits = sizeof(T) * 8;
  LOG(INFO) << "Nbits " << nbits;

  for (size_t i = 0; i < a.size(); ++i) {
    T x = a[i] ^ b[i];

    for (size_t j = 0; j < nbits; ++j) {
      if (x & 1) {
        ++diff;
      }
      x >>= 1;
    }
  }

  return diff;
}

TEST(HammingDistance, Empty) { EXPECT_EQ(HammingDistance<int>({}, {}), 0); }

TEST(HammingDistance, UnequalLengths) {
  EXPECT_EQ(HammingDistance<int>({1}, {}), -1);
  EXPECT_EQ(HammingDistance<int>({}, {-1}), -1);
}

TEST(HammingDistance, OneMatchingInt) {
  EXPECT_EQ(HammingDistance<int>({1}, {1}), 0);
}

TEST(HammingDistance, OneUnmatchingInt) {
  EXPECT_EQ(HammingDistance<int>({1}, {0}), 1);
  EXPECT_EQ(HammingDistance<int>({2}, {1}), 2);
}

TEST(HammingDistance, TwoUnmatchingInts) {
  EXPECT_EQ(HammingDistance<int>({1, 1}, {0, 0}), 2);
  EXPECT_EQ(HammingDistance<int>({2, 1}, {1, 1}), 2);
}

TEST_MAIN();
