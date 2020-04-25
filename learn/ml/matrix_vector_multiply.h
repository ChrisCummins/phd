#pragma once

#include <array>

namespace ml {

template <typename T, size_t X, size_t H>
std::array<T, X> MatrixVectorMultiply(const std::array<T, X * H>& A,
                                      const std::array<T, X>& B) {
  std::array<T, X> out;

  for (size_t i = 0; i < X; ++i) {
    out[i] = 0;
    for (size_t j = 0; j < H; ++j) {
      out[i] += A[j * H + i] * B[j];
    }
  }

  return out;
}

}  // namespace ml
