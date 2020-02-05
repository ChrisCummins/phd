#pragma once

#include <array>
#include <tuple>

#include "learn/ml/elementwise_addition.h"
#include "learn/ml/matrix_vector_multiply.h"
#include "learn/ml/sigmoid.h"
#include "learn/ml/softmax.h"

namespace ml {

template <typename T, size_t X, size_t H>
std::tuple<std::array<T, X>, std::array<T, X>, std::array<T, X>,
           std::array<T, X> >
FeedForward(const std::array<T, X * H>& W1, const std::array<T, X * H>& W2,
            const std::array<T, X>& b1, const std::array<T, X>& b2,
            const std::array<T, X>& input) {
  // Z1 = W1.dot(x) + b1;
  std::array<T, X> Z1 = MatrixVectorMultiply<float, X, H>(W1, input);
  ElementwiseAddition(b1, &Z1);

  // A1 = sigmoid(Z1)
  std::array<T, X> A1 = Sigmoid(Z1);

  // Z2 = W2.dot(A1) + b2
  std::array<T, X> Z2 = MatrixVectorMultiply<float, X, H>(W2, A1);
  ElementwiseAddition(b2, &Z2);

  // A2 = softmax(Z2)
  std::array<T, X> A2 = Softmax(Z2);

  return {Z1, A1, Z2, A2};
}

}  // namespace ml
