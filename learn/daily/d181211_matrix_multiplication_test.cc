#include "learn/daily/d181211_matrix_multiplication.h"

#include "labm8/cpp/test.h"

namespace labm8 {
namespace learn {
namespace {

TEST(ColumnVector, ElementCount) {
  Matrix a(2, 3);

  auto column = ColumnVector(a, 0);
  EXPECT_EQ(column.size(), 2);
}

TEST(ColumnVector, Values) {
  Matrix a(2, 3);
  a(0, 0) = 1;
  a(0, 1) = 2;
  a(0, 2) = 3;
  a(1, 0) = 4;
  a(1, 1) = 5;
  a(1, 2) = 6;

  auto column = ColumnVector(a, 0);
  EXPECT_EQ(column[0], 1);
  EXPECT_EQ(column[1], 4);

  auto column_b = ColumnVector(a, 1);
  EXPECT_EQ(column_b[0], 2);
  EXPECT_EQ(column_b[1], 5);
}

TEST(ColumnVector, OutOfBoundsCheck) {
  Matrix a(2, 3);

  ASSERT_DEATH(ColumnVector(a, -1), "CHECK");
  ASSERT_DEATH(ColumnVector(a, 3), "CHECK");
}

TEST(RowVector, ElementCount) {
  Matrix a(2, 3);

  auto row = RowVector(a, 0);
  EXPECT_EQ(row.size(), 3);
}

TEST(RowVector, Values) {
  Matrix a(2, 3);
  a(0, 0) = 1;
  a(0, 1) = 2;
  a(0, 2) = 3;
  a(1, 0) = 4;
  a(1, 1) = 5;
  a(1, 2) = 6;

  auto row = RowVector(a, 0);
  EXPECT_EQ(row[0], 1);
  EXPECT_EQ(row[1], 2);
  EXPECT_EQ(row[2], 3);

  auto row_b = RowVector(a, 1);
  EXPECT_EQ(row_b[0], 4);
  EXPECT_EQ(row_b[1], 5);
  EXPECT_EQ(row_b[2], 6);
}

TEST(RowVector, OutOfBoundsCheck) {
  Matrix a(2, 3);

  ASSERT_DEATH(RowVector(a, -1), "CHECK");
  ASSERT_DEATH(RowVector(a, 2), "CHECK");
}

TEST(DotProduct, InvalidInputSizes) {
  Vector a(2);
  Vector b(3);

  ASSERT_DEATH(DotProduct(a, b), "CHECK");
}

TEST(DotProduct, Values) {
  Vector a{0, 1, 2};
  Vector b{3, 4, 5};

  EXPECT_EQ(DotProduct(a, b), 14);  // 0 * 3 + 1 * 4 + 2 * 5
  EXPECT_EQ(DotProduct(a, a), 5);   // 0 * 0 + 1 * 1 + 2 * 2
  EXPECT_EQ(DotProduct(b, b), 50);  // 3 * 3 + 4 * 4 + 5 * 5
}

TEST(MatrixMultiplication, OutputRowCount) {
  Matrix a(2, 3);
  Matrix b(3, 2);

  Matrix c = Multiply(a, b);
  EXPECT_EQ(c.size1(), 2);
}

TEST(MatrixMultiplication, OutputColumnCount) {
  Matrix a(2, 3);
  Matrix b(3, 2);

  Matrix c = Multiply(a, b);
  EXPECT_EQ(c.size2(), 2);
}

TEST(MatrixMultiplication, OutputValues) {
  Matrix a(2, 3);
  a(0, 0) = 1;
  a(0, 1) = 2;
  a(0, 2) = 3;
  a(1, 0) = 4;
  a(1, 1) = 5;
  a(1, 2) = 6;

  Matrix b(3, 2);
  b(0, 0) = 7;
  b(0, 1) = 8;
  b(1, 0) = 9;
  b(1, 1) = 10;
  b(2, 0) = 11;
  b(2, 1) = 12;

  Matrix c = Multiply(a, b);
  EXPECT_EQ(c(0, 0), 58);   // 1 * 7 + 2 * 9 + 3 * 11
  EXPECT_EQ(c(0, 1), 64);   // 1 * 8 + 2 * 10 + 3 * 12
  EXPECT_EQ(c(1, 0), 139);  // 4 * 7 + 5 * 9 + 6 * 11
  EXPECT_EQ(c(1, 1), 154);  // 4 * 8 + 5 * 10 + 6 * 12
}

}  // namespace
}  // namespace learn
}  // namespace labm8

TEST_MAIN();
