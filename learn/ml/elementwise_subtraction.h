#pragma once

#include <array>

namespace ml {

// Compute elementwise A - B and return a new array.
template <typename T, size_t N>
std::array<T, N> ElementwiseSubtract(const std::array<T, N>& A,
                                     const std::array<T, N>& B) {
  std::array<T, N> out;
  for (size_t i = 0; i < N; ++i) {
    out[i] = A[i] - B[i];
  }

  return out;
}

}  // namespace ml
