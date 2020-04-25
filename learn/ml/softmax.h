#pragma once

#include <array>
#include <cmath>

namespace ml {

// Time: O(n)
// Space: O(n)
template <typename T, size_t n>
std::array<T, n> Softmax(std::array<T, n>& X) {
  std::array<T, n> out;
  T sum = 0;

  // Offset calculation by max value to reduce risk of overflow when input
  // contains large values. This bounds the input to exp() in the range
  // [-inf,0].
  const T max = *std::max_element(X.begin(), X.end());

  for (size_t i = 0; i < n; ++i) {
    out[i] = exp(X[i] - max);
    sum += out[i];
  }

  for (size_t i = 0; i < n; ++i) {
    out[i] /= sum;
  }

  return out;
}

}  // namespace ml
