#include <lm/lm>

#include "./tests.hpp"

TEST(Lm, Vector) {
  lm::Vector<> v{1, 2, 3, 4, 5};

  auto v2 = v + 1;
  v += 1;

  lm::Vector<> v1{2, 3, 4, 5, 6};

  ASSERT_TRUE(v == v1);
  ASSERT_TRUE(v2 == v1);
}

TEST(Lm, Matrix) {
  lm::Matrix<> m{10, 5};
  lm::Matrix<> m1{{1, 2, 3}, {4, 5, 6}};

  try {
    lm::Matrix<> m2{{1, 2, 3}, {4}};
    FAIL();
  } catch (const std::runtime_error &) {
  }

  std::cout << m1;
}

TEST_MAIN();
