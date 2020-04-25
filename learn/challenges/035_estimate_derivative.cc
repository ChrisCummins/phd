// Write a function to estimate the derivative of a function at a point.
#include "labm8/cpp/test.h"

#include <functional>

template <typename T>
T ComputeDerivative(std::function<T(const T&)> fn, const T& x) {
  const T h = 0.00001;
  return (fn(x + h) - fn(x)) / h;
}

TEST(ComputeDerivative, XSquaredDouble) {
  const auto fn = [&](const double& x) { return x * x; };
  EXPECT_NEAR(ComputeDerivative<double>(fn, 0), 0, 0.001);
  EXPECT_NEAR(ComputeDerivative<double>(fn, 1), 2, 0.001);
  EXPECT_NEAR(ComputeDerivative<double>(fn, 2), 4, 0.001);
}

TEST(ComputeDerivative, PolynomialDouble) {
  const auto fn = [&](const double& x) { return (x * x * x) + 3; };
  EXPECT_NEAR(ComputeDerivative<double>(fn, 0), 0, 0.001);
  EXPECT_NEAR(ComputeDerivative<double>(fn, 1), 3, 0.001);
  EXPECT_NEAR(ComputeDerivative<double>(fn, 2), 12, 0.001);
}

TEST(ComputeDerivative, AnotherPolynomial) {
  const auto fn = [&](const double& x) { return 10 - 2 * x; };
  EXPECT_NEAR(ComputeDerivative<double>(fn, 0), -2, 0.001);
  EXPECT_NEAR(ComputeDerivative<double>(fn, 1), -2, 0.001);
  EXPECT_NEAR(ComputeDerivative<double>(fn, 2), -2, 0.001);
}

TEST(ComputeDerivative, XSquaredFloat) {
  const auto fn = [&](const float& x) { return x * x; };
  EXPECT_NEAR(ComputeDerivative<float>(fn, 0), 0, 0.01);
  EXPECT_NEAR(ComputeDerivative<float>(fn, 1), 2, 0.01);
  EXPECT_NEAR(ComputeDerivative<float>(fn, 2), 4, 0.01);
}

TEST_MAIN();
