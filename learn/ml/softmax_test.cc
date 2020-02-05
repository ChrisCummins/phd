#include "learn/ml/softmax.h"
#include <array>
#include <cmath>
#include "labm8/cpp/test.h"

namespace ml {

TEST(Softmax, Sum) {
  std::array<float, 3> Xin({1, 2, 3});

  const auto Xout = Softmax(Xin);

  float sum = 0;
  for (auto x : Xout) {
    sum += x;
  }

  EXPECT_NEAR(sum, 1.0, 0.001);
}

TEST(Softmax, Values) {
  std::array<float, 3> Xin({1, 2, 3});

  const auto Xout = Softmax(Xin);

  EXPECT_NEAR(Xout[0], 0.0900, 0.001);
  EXPECT_NEAR(Xout[1], 0.2447, 0.001);
  EXPECT_NEAR(Xout[2], 0.6652, 0.001);
}

}  // namespace ml

TEST_MAIN();
