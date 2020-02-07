#include <vector>
#include "labm8/cpp/test.h"

uint64_t Factorial(const uint64_t n) {
  uint64_t x = 1;
  for (size_t i = 2; i <= n; i++) {
    x *= i;
  }
  return x;
}

TEST(NChooseK, FactorialHeler) {
  EXPECT_EQ(Factorial(0), 1);
  EXPECT_EQ(Factorial(1), 1);
  EXPECT_EQ(Factorial(2), 2);
  EXPECT_EQ(Factorial(3), 6);
  EXPECT_EQ(Factorial(4), 24);
  EXPECT_EQ(Factorial(5), 120);
}

uint64_t NChooseKFactorial(const uint64_t n, const uint64_t k) {
  return Factorial(n) / (Factorial(k) * Factorial(n - k));
}

TEST(NChooseK, Factorial) {
  EXPECT_EQ(NChooseKFactorial(5, 0), 1);
  EXPECT_EQ(NChooseKFactorial(5, 1), 5);
  EXPECT_EQ(NChooseKFactorial(5, 2), 10);
  EXPECT_EQ(NChooseKFactorial(5, 3), 10);
  EXPECT_EQ(NChooseKFactorial(5, 4), 5);
  EXPECT_EQ(NChooseKFactorial(5, 5), 1);

  EXPECT_EQ(NChooseKFactorial(10, 0), 1);
  EXPECT_EQ(NChooseKFactorial(10, 1), 10);
  EXPECT_EQ(NChooseKFactorial(10, 2), 45);
  EXPECT_EQ(NChooseKFactorial(10, 3), 120);
  EXPECT_EQ(NChooseKFactorial(10, 4), 210);
  EXPECT_EQ(NChooseKFactorial(10, 5), 252);
  EXPECT_EQ(NChooseKFactorial(10, 6), 210);
  EXPECT_EQ(NChooseKFactorial(10, 7), 120);
  EXPECT_EQ(NChooseKFactorial(10, 8), 45);
  EXPECT_EQ(NChooseKFactorial(10, 9), 10);
  EXPECT_EQ(NChooseKFactorial(10, 10), 1);
}

uint64_t NChooseKRecursive(const uint64_t n, const uint64_t k) {
  if (k == 0 || k == n) {
    return 1;
  }
  return NChooseKRecursive(n - 1, k - 1) + NChooseKRecursive(n - 1, k);
}

TEST(NChooseK, Recursive) {
  EXPECT_EQ(NChooseKRecursive(5, 0), 1);
  EXPECT_EQ(NChooseKRecursive(5, 1), 5);
  EXPECT_EQ(NChooseKRecursive(5, 2), 10);
  EXPECT_EQ(NChooseKRecursive(5, 3), 10);
  EXPECT_EQ(NChooseKRecursive(5, 4), 5);
  EXPECT_EQ(NChooseKRecursive(5, 5), 1);

  EXPECT_EQ(NChooseKRecursive(10, 0), 1);
  EXPECT_EQ(NChooseKRecursive(10, 1), 10);
  EXPECT_EQ(NChooseKRecursive(10, 2), 45);
  EXPECT_EQ(NChooseKRecursive(10, 3), 120);
  EXPECT_EQ(NChooseKRecursive(10, 4), 210);
  EXPECT_EQ(NChooseKRecursive(10, 5), 252);
  EXPECT_EQ(NChooseKRecursive(10, 6), 210);
  EXPECT_EQ(NChooseKRecursive(10, 7), 120);
  EXPECT_EQ(NChooseKRecursive(10, 8), 45);
  EXPECT_EQ(NChooseKRecursive(10, 9), 10);
  EXPECT_EQ(NChooseKRecursive(10, 10), 1);
}

TEST_MAIN();
