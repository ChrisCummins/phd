// Implement Gradient Descent.

#include <cmath>
#include <functional>
#include <sstream>
#include <vector>
#include "labm8/cpp/logging.h"
#include "labm8/cpp/test.h"

using std::stringstream;
using std::vector;

float EstimateGradient(const std::function<double(const vector<float>&)>& fn,
                       size_t i, vector<float>* X, const float& h = 0.0001) {
  float y = fn(*X);

  (*X)[i] += h;
  float yh = fn(*X);
  (*X)[i] -= h;

  return (yh - y) / h;
}

// Estimate the gradient of fn(X) using offset h.
vector<float> EstimateGradients(
    const std::function<double(const vector<float>&)>& fn,
    const vector<float>& X, const float& h = 0.0001) {
  vector<float> gradients;
  gradients.reserve(X.size());

  vector<float> Xh = X;
  for (size_t i = 0; i < X.size(); ++i) {
    gradients.push_back(EstimateGradient(fn, i, &Xh, h));
  }

  return gradients;
}

vector<float> GradientDescent(
    const std::function<double(const vector<float>&)>& fn,
    const vector<float>& start, const float& stepSize = 0.001,
    const float& maxL2Norm = 0.001, const size_t& maxIter = 100000) {
  vector<float> val = start;

  float l2Norm = 0;
  for (size_t step = 1; step <= maxIter; ++step) {
    l2Norm = 0;
    auto gradients = EstimateGradients(fn, val);
    for (size_t i = 0; i < gradients.size(); ++i) {
      val[i] -= gradients[i] * stepSize;
      l2Norm += val[i] * val[i];
    }

    l2Norm = sqrt(l2Norm);

    if (step % 100 == 0) {
      LOG(INFO) << "Step " << step << " L2-norm " << l2Norm;
    }

    if (l2Norm < maxL2Norm) {
      break;
    }
  }

  stringstream ss;
  for (size_t i = 0; i < val.size(); ++i) {
    ss << val[i] << " ";
  }
  LOG(INFO) << "Terminated with l2Norm(f(X)) = " << l2Norm << " at X = [ "
            << ss.str() << "]";

  return val;
}

// f(x) = max(x, 0)
TEST(GradientDescent, ReLuSingleVar) {
  vector<float> start = {5.};

  auto minX = GradientDescent(
      [&](const vector<float>& X) { return X[0] < 0 ? 0 : X[0]; }, start);

  ASSERT_EQ(minX.size(), 1);
  EXPECT_NEAR(minX[0], 0, 0.001);
}

// f(x) = sum(x_i ^ 2)
TEST(GradientDescent, SumSquare) {
  vector<float> start = {1., 2., 3., 4.};

  auto minX = GradientDescent(
      [&](const vector<float>& X) {
        float y = 0;
        for (const auto& x : X) {
          y += x * x;
        }
        return y;
      },
      start);

  ASSERT_EQ(minX.size(), 4);
  EXPECT_NEAR(minX[0], 0, 0.001);
}

TEST_MAIN();
