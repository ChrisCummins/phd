// This is a C++ implementation of Zhenye's blog post on
// implementing a neural network from scratch using numpy:
//
// https://zhenye-na.github.io/2018/09/09/build-neural-network-with-mnist-from-scratch.html

#include "labm8/cpp/app.h"
#include "labm8/cpp/logging.h"

#include "datasets/mnist/cpp/mnist_data_loader.h"

#include "learn/ml/elementwise_addition.h"
#include "learn/ml/feed_forward.h"
#include "learn/ml/matrix_vector_multiply.h"
#include "learn/ml/randomness.h"
#include "learn/ml/softmax.h"

static const char* usage = "Unpack mnist data files.";

using namespace ml;

// Dimensionality of our input.
static const size_t X = 28 * 28;

// The size of our hidden layers.
static const size_t H = 64;

int main(int argc, char** argv) {
  labm8::InitApp(&argc, &argv, usage);

  if (argc != 1) {
    LOG(ERROR) << "Unexpected arguments";
    return 1;
  }

  // Load the MNIST dataset.
  auto data = mnist::MnistDataLoader().Load().ValueOrDie();
  LOG(INFO) << "Loaded dataset";

  // Weights, initialized randomly with variance 1.
  std::array<float, H * X> W1;
  std::array<float, H * X> W2;
  ArrayFillRandVarOne<float, H * X, X>(&W1);
  ArrayFillRandVarOne<float, H * X, X>(&W2);

  // Biases, initialized to zero.
  std::array<float, X> b1;
  std::array<float, X> b2;
  b1.fill(0);
  b2.fill(0);

  std::array<float, 28 * 28> input;

  FeedForward<float, X, H>(W1, W2, b1, b2, input);

  // This is work-in-progress. Pick it up from here ...

  return 0;
}
