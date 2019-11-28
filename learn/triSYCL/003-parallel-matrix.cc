#include "./003-parallel-matrix.h"

#include <iostream>
#include <vector>

///////////
// Tests //
///////////

const auto test_a = matrix({{1, 2, 3}, {4, 5, 6}});
const auto test_b = matrix({{4, 5, 6}, {1, 2, 3}});

const auto test_add = matrix({{5, 7, 9}, {5, 7, 9}});
const auto test_sub = matrix({{-3, -3, -3}, {3, 3, 3}});

TEST(matrix, addition) { ASSERT_TRUE(test_a + test_b == test_add); }

TEST(matrix, subtraction) { ASSERT_TRUE(test_a - test_b == test_sub); }

PHD_MAIN();
