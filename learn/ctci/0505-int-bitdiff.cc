/*
 * Write a function to determine the number of bits you would need to
 * flip to convert integer A to integer B.
 *
 * Example: Input 29 (or: 11101), 15 (or: 01111), Output: 2
 */
#include "./ctci.h"

//
// My solution. Count the number of bits set in the XOR of the two
// values. Template solution supports any type which supports bitwise
// XOR operator.
//
template <typename T>
unsigned int bitdiff(const T& a, const T& b) {
  T diff{a ^ b};
  unsigned int count{0};

  for (size_t i = 0; i < sizeof(T) * 8; ++i)
    if (diff & (1 << i)) ++count;

  return count;
}

///////////
// Tests //
///////////

TEST(BitDiff, bitdiff) {
  ASSERT_EQ(0u, bitdiff(10, 10));
  ASSERT_EQ(2u, bitdiff(29, 15));
  ASSERT_EQ(1u, bitdiff(0, 1));
  ASSERT_EQ(2u, bitdiff(1, 2));
}

CTCI_MAIN();
