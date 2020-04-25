#pragma once

#include <array>
#include <tuple>

#include "learn/ml/elementwise_subtraction.h"

using std::array;
using std::tuple;

namespace ml {

template <typename T, size_t X, size_t H>
tuple<array<T, X * H>, array<T, X * H>, array<T, X>, array<T, X>, >
BackPropagate(const array<T, X * H>& W1, const array<T, X>& b1,
              const array<T, X * H>& W2, const array<T, X>& b2,
              const array<T, X>& Z1, const array<T, X>& A1,
              const array<T, X>& A2, const array<T, X>& input,
              const array<T, X>& output, size_t batchSize, ) {
  // Begin at last error.
  array<T, X> dZ2 = ElementwiseSubtract(A2, Y);

  // Compute gradients at last layer.
  array<T, X> dW2 = MatrixScalarMultiply(
      1 / batchSize, MatrixVectorMultiply(dZ2, Transpose(A1)));
  array<T, X> db2 =
      ScalarVectorMultiplication(1 / batchSize, MatrixRowSum(dZ2));

  // Back propagate through the first layer.
  array<T, X> dA1 = MatrixVectorMultiply(Transpose(W2), dZ2);
  array<T, X> dZ1 = dA1 * Sigmoid(Z1) * (1 - Sigmoid(Z1));

  // Compute gradients at first layer.
  array<T, X> dW1 = MatrixScalarMultiply(
      1 / batchSize, MatrixVectorMultiply(dZ1, Transpose(input)));
  array<T, X> db1 =
      ScalarVectorMultiplication(1 / batchSize, MatrixRowSum(dZ1));

  return { dW1, db1, dW2, db2 }
}

}  // namespace ml
