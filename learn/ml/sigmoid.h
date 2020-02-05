#pragma once

#include <array>
#include <cmath>

namespace ml {

// Time: O(n)
// Space: O(n)
template <typename T, size_t N>
std::array<T, N> Sigmoid(const std::array<T, N>& in) {
  std::array<T, N> out;

  T sum = 0;
  for (size_t i = 0; i < N; ++i) {
    out[i] = exp(in[i]);
    sum += out[i];
  }

  for (size_t i = 0; i < N; ++i) {
    out[i] /= sum;
  }

  return out;
}

}  // namespace ml
